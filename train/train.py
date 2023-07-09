import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import copy
from sklearn import metrics
import numpy as np
from pathlib import Path
import logging
import json

import torch
torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from network import model_provider
from data_provider.base_provider import BaseProvider
from utils import resume_ckpt,AverageMeter,Params,save_dir_root,save_dict_to_json
from base_process import BaseProcess

#Note: In this project, use tow kinds of models. The directory tree used for saving model files is run/ckpt/{modelName}/run_{id},
# where `modelName` is either `C3D` or `VisualRNNModel` and `id` represents the number of times of training. And the name
# of model file is `epoch-{epoch}.pth.tar`. Use tensorboard for visualization, and the directory tree used for saving event files is run/logs/{modelName}/run_{id}


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params=Params(dict_params={
    'num_epochs':20,
    'resume_epoch':0,
    'useTest':True,
    'test_interval':5,
    'save_epoch':10,
    'lr':1e-4,
    'batch_size':5,
    'modelName':'VisualRNNModel',# C3D/VisualRNNModel
    'visualModelName':'vgg11',# None/vgg11
    'pretrained':True,
    'num_classes':4,
    'dataset':'stroke_dataset',
    'seed':111,
    'weighted_loss':False,# decide whether to assign different weights to different classes during calculating loss values
    'sampling_strategy': None,
    'seq_length':16,
    'crop_size':112,
    'resize_width_height':[171,128]

})


def train_model():
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    # get the model
    model=model_provider.get_model(params.modelName,num_classes=params.num_classes,pretrained=params.pretrained,visualModelName=params.visualModelName)

    train_params = [{'params': model.get_1x_lr_params(), 'lr': params.lr},
                        {'params': model.get_10x_lr_params(), 'lr': params.lr * 10}]#列表，第一个元素为C3D网络所有卷积层及前两个连接层的所有参数及学习率lr；第二个元素为最后一个全连接层的参数及10*lr（这里学习率的区别是由于fc8模块需要从头开始训练，参数调整幅度不宜过小，而除fc8模块外的其它模块均采用预训练模型的参数，此时对他们的参数只需要进行细微调整

    train_dataloader = DataLoader(
        BaseProvider(split='train', resize_width_height=params.resize_width_height, modelName=params.modelName,
                     crop_size=params.crop_size, seq_length=params.seq_length,
                     sampling_strategy=params.sampling_strategy), batch_size=params.batch_size, shuffle=True,
        num_workers=1)  # 随机采样
    val_dataloader = DataLoader(
        BaseProvider(split='val', resize_width_height=params.resize_width_height, modelName=params.modelName,
                     crop_size=params.crop_size, seq_length=params.seq_length), batch_size=params.batch_size,
        num_workers=1)  # 顺序采样
    test_dataloader = DataLoader(
        BaseProvider(split='test', resize_width_height=params.resize_width_height, modelName=params.modelName,
                     crop_size=params.crop_size, seq_length=params.seq_length), batch_size=params.batch_size,
        num_workers=1)  # 顺序采样

    weight= train_dataloader.dataset.loss_weights()   if params.weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight)  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=params.lr, momentum=0.9, weight_decay=5e-4)#使用动量和L2正则化的随机梯度下降
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs 每间隔10个epoch将学习率除以10进行衰减（实际是每运行10次scheduler.step就衰减lr，真正间隔多少个epoch还取决于scheduler.step与optimizer.step的相对位置

    # model, optimizer, scheduler, run_id=resume_ckpt(model,optimizer,scheduler,params.resume_epoch,params.modelName,params.visualModelName)
    model, run_id=resume_ckpt(model,params.resume_epoch,params.modelName,params.visualModelName)

    if params.modelName=='C3D':
        logger_dir=Path(os.path.join(save_dir_root,'logger',params.modelName,'run_'+str(run_id)))
        save_dir = Path(os.path.join(save_dir_root, 'run', 'ckpt', params.modelName,'run_' + str(run_id)))  # the path of folder to store model files
        log_dir = os.path.join(save_dir_root, 'run', 'logs', params.modelName,'run_' + str(run_id))  # The folder doesn't need to exist in advance
        metric_dir=Path(os.path.join(save_dir_root,'run','metricsValue',params.modelName,'run_'+str(run_id)))
    else:
        logger_dir=Path(os.path.join(save_dir_root,'logger',params.modelName,params.visualModelName,'run_'+str(run_id)))
        save_dir = Path(os.path.join(save_dir_root, 'run', 'ckpt', params.modelName,params.visualModelName,'run_' + str(run_id)))  # the path of folder to store model files
        log_dir = os.path.join(save_dir_root, 'run', 'logs', params.modelName,params.visualModelName,'run_' + str(run_id))  # The folder doesn't need to exist in advance
        metric_dir=Path(os.path.join(save_dir_root,'run','metricsValue',params.modelName,params.visualModelName,'run_'+str(run_id)))


    logger_dir.mkdir(parents=True,exist_ok=True)# `logger_dir` must exist before calling `set_logger`
    BaseProcess.set_logger(str(logger_dir/'training.log'))

    logging.info('parameters setting: %s',json.dumps(params.dict))

    logging.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))# logging the total number of parameters of the model and the output unit is million(M)

    model.to(device)#Moves all model parameters and buffers to the device.
    criterion.to(device)

    writer = SummaryWriter(log_dir=log_dir)#tensorboard使用


    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}


    best_acc_model_wts=best_ba_model_wts=copy.deepcopy(model.state_dict())#存储相应性能最好的模型的参数
    best_acc_model_epoch=best_ba_model_epoch=0
    best_acc=best_ba=0.0



    for epoch in range(params.resume_epoch, params.num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = AverageMeter()
            running_acc = AverageMeter()
            running_balanced_accuracy=AverageMeter()
            running_auc=AverageMeter()

            # set model to train() or eval() mode depending on whether it is trained or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()#https://zhuanlan.zhihu.com/p/136902153
                model.train() #设置self.training=True，启用dropout和batchNorm层在train时的功能 https://blog.csdn.net/qq_52852138/article/details/123769937
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):#tqdm进度条加载模块，inputs指每个batch的样本
                # move inputs and labels to the device the training is taking place on

                inputs = Variable(inputs, requires_grad=True).to(device)
                # raise ValueError("break down")
                labels=labels.long()
                labels = Variable(labels).to(device)
                optimizer.zero_grad()#梯度置零

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                if params.modelName=='VisualRNNModel':
                    outputs=outputs[:,-1,:]

                probs = nn.Softmax(dim=1)(outputs)#得到softmax处理后的值，shape为[10,4]
                preds = torch.max(probs, 1)[1]#预测的label所在索引位置，shape为10
                loss = criterion(outputs, labels)#不能将probs值替代outputs值，原因是CrossEntropyLoss函数内部有实现softmax  多个样本损失值的平均值

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                np_preds=preds.detach().cpu().numpy()
                np_labels=labels.detach().cpu().numpy()
                np_probs=probs.detach().cpu().numpy()
                running_loss.update(loss.item(), inputs.size(0)) #对应epoch的损失值总和  使用loss.item()的原因：https://blog.csdn.net/Viviane_2022/article/details/128379670
                running_acc.update(np.sum(np_preds == np_labels)/inputs.size(0),inputs.size(0)) #每个epoch被正确分类的视频数
                running_balanced_accuracy.update(metrics.balanced_accuracy_score(np_labels,np_preds),inputs.size(0))    # the average of recall obtained on each label
                running_auc.update(metrics.roc_auc_score(np_labels, np_probs, average='macro', multi_class='ovo',labels=[0,1,2,3]),inputs.size(0))

            trainval_loaders[phase].dataset.reset()

            #deep copy the model
            if phase=='val' :
                if running_acc.avg>best_acc:
                    best_acc=running_acc.avg
                    best_acc_model_epoch=epoch
                    best_acc_model_wts=copy.deepcopy(model.state_dict())
                if running_balanced_accuracy.avg>best_ba:
                    best_ba=running_balanced_accuracy.avg
                    best_ba_model_epoch = epoch
                    best_ba_model_wts = copy.deepcopy(model.state_dict())
                # save the metrics obtained from every epoch to the `metrics.json` file
                epoch_acc_auc_ba={'epoch':epoch,'acc':running_acc.avg,'auc':running_auc.avg,'balanced_accuracy':running_balanced_accuracy.avg}
                metric_dir.mkdir(parents=True,exist_ok=True)
                save_dict_to_json(epoch_acc_auc_ba,str(metric_dir/'metrics.json'))

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', running_loss.avg, epoch)
                writer.add_scalar('data/train_acc_epoch', running_acc.avg, epoch)
                writer.add_scalar('data/train_ba_epoch', running_balanced_accuracy.avg, epoch)
                writer.add_scalar('data/train_auc_epoch', running_auc.avg, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', running_loss.avg, epoch)
                writer.add_scalar('data/val_acc_epoch', running_acc.avg, epoch)
                writer.add_scalar('data/val_ba_epoch', running_balanced_accuracy.avg, epoch)
                writer.add_scalar('data/val_auc_epoch', running_auc.avg, epoch)

            logging.info("[{}] Epoch: {}/{} Loss: {} Acc: {} Auc: {} Balanced_accuracy: {}".format(phase, epoch+1, params.num_epochs, running_loss.avg,running_acc.avg,running_auc.avg,running_balanced_accuracy.avg))
            stop_time = timeit.default_timer()
            logging.info("Execution time: " + str(stop_time - start_time) + "\n")#train/val一个epoch所需时间

        save_dir.mkdir(parents=True, exist_ok=True)
        if epoch % params.save_epoch == (params.save_epoch - 1):#是否需要保存权重文件
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                # 'opt_dict': optimizer.state_dict(),
                # 'sch_dict':scheduler.state_dict(),##scheduler的状态
            }, os.path.join(save_dir,'epoch-' + str(epoch) + '.pth.tar'))
            logging.info("Save model at {}\n".format(os.path.join(save_dir,'epoch-' + str(epoch) + '.pth.tar')))#模型保存路径：save_dir+"models"+saveName+"_epoch_"+str(epoch)+".pth.tar"

        if params.useTest and epoch % params.test_interval == (params.test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = AverageMeter()
            running_acc = AverageMeter()
            running_balanced_accuracy = AverageMeter()
            running_auc = AverageMeter()

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels=labels.long()
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                if params.modelName == 'VisualRNNModel':
                    outputs = outputs[:, -1, :]

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                np_preds = preds.detach().cpu().numpy()
                np_labels = labels.detach().cpu().numpy()
                np_probs = probs.detach().cpu().numpy()
                running_loss.update(loss.item(), inputs.size(0)) #对应epoch的损失值总和  使用loss.item()的原因：https://blog.csdn.net/Viviane_2022/article/details/128379670
                running_acc.update(np.sum(np_preds == np_labels)/inputs.size(0),inputs.size(0)) #每个epoch被正确分类的视频数
                running_balanced_accuracy.update(metrics.balanced_accuracy_score(np_labels,np_preds),inputs.size(0))    # the average of recall obtained on each label
                running_auc.update(metrics.roc_auc_score(np_labels, np_probs, average='macro', multi_class='ovo',labels=[0,1,2,3]),inputs.size(0))

            test_dataloader.dataset.reset()# Reset for next epoch

            writer.add_scalar('data/test_loss_epoch', running_loss.avg, epoch)
            writer.add_scalar('data/test_acc_epoch', running_acc.avg, epoch)
            writer.add_scalar('data/test_ba_epoch', running_balanced_accuracy.avg, epoch)
            writer.add_scalar('data/test_auc_epoch', running_auc.avg, epoch)

            logging.info("[test] Epoch: {}/{} Loss: {} Acc: {} Auc: {} Balanced_accuracy: {}".format(epoch+1, params.num_epochs, running_loss.avg,running_acc.avg,running_auc.avg,running_balanced_accuracy.avg))
            stop_time = timeit.default_timer()
            logging.info("Execution time: " + str(stop_time - start_time) + "\n")

    ##store the parameters and buffers of the best model
    torch.save({'state_dict':best_acc_model_wts,'epoch':best_acc_model_epoch},os.path.join(save_dir,'best_acc_model.pth.tar'))
    torch.save({'state_dict':best_ba_model_wts,'epoch':best_ba_model_epoch},os.path.join(save_dir,'best_ba_model.pth.tar'))


    writer.close()


if __name__ == "__main__":
    train_model()
    # train_dataloader = DataLoader(BaseProvider(split='train',resize_width_height=params.resize_width_height,modelName=params.modelName,crop_size=params.crop_size,seq_length=params.seq_length,sampling_strategy=params.sampling_strategy), batch_size=params.batch_size, shuffle=True, num_workers=1)#随机采样
    # weight=train_dataloader.dataset.loss_weights()
    # print(weight)
    # d=params.dict
    # print(d)
    # print(type(d))

    # args=parse_args()

    # train_model(args,args.dataset)