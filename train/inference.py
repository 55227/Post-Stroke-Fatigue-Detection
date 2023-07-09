#zmq
import numpy as np
import cv2
import os
from pathlib import Path
import pandas as pd
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn import metrics


from network import model_provider
from data_provider.test_provider import TestProvider,normalize_C3D,normalize_RNN
from utils import Params

# save_dir_root=r'C:\Users\dell\Desktop\run_autodl'
dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))# the directory where this file is located
save_dir_root=os.path.join(dir_root,'run')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params=Params(dict_params={
    'batch_size':5,
    'visualModelName':'None',# None/vgg11
    'eval':'all',# best_acc/best_ba/all: decide what type of trained models to inference
    'num_classes':4,
    'seq_length':16,
    'crop_size':112,
    'resize_width_height':[171,128],
    'model_weights':[0.2,0.25,0.25,0.1,0.1,0.1]# assign different weitghs to different models to perform a weighted sum

})
def main():
    ensembleModelNames=['C3D']
    test_dataset=TestProvider()
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size)  # 顺序采样

    ensemble_model_files=[]# dict of trained model files. {'C3D':[paths of corresponding model files],'VisualRNNModel':{'resnet101':[],'vgg11':[]}}
    ensemble_models=[]# list of all trained models

    # visualModelPath=r"C:\Users\dell\Desktop\run_autodl\ckpt\VisualRNNModel\vgg11\run_1\best_acc_model.pth.tar"
    visualModelPath=os.path.join(save_dir_root,'ckpt','VisualRNNModel','vgg11','run_0','best_acc_model.pth.tar')
    visualModel=None

    for modelname in ensembleModelNames:
        model_dir = os.path.join(save_dir_root, 'ckpt', modelname)
        if params.eval == 'best_acc':
            ensemble_model_files=list(Path(model_dir).glob("**/best_acc_model.pth.tar"))# list of trained model files using `modelname` model
        elif params.eval == 'epoch-9':
            ensemble_model_files=list(Path(model_dir).glob("**/epoch-9.pth.tar"))# list of trained model files using `modelname` model
        elif params.eval == 'epoch-19':
            ensemble_model_files=list(Path(model_dir).glob("**/epoch-19.pth.tar"))# list of trained model files using `modelname` model
        elif params.eval=='all':
            ensemble_model_files=list(Path(model_dir).glob("**/*.pth.tar"))

    c3d_index=0# 使用原C3D得到的最佳准确率模型在ensemble_models中的索引
    resample_index=2# 使用基于重采样的C3D得到的最佳准确率模型在ensemble_models中的索引



    print(ensemble_model_files)


    for model_file in ensemble_model_files:
        model_class=model_provider.get_model(modelname)
        model_class=model_class.to(device)
        state=torch.load(model_file)
        # state_dict = state['state_dict']
        model_class.load_state_dict(state['state_dict'])
        model_class.eval()
        ensemble_models.append(model_class)

    if visualModelPath:
        visualModel = model_provider.get_model('VisualRNNModel', visualModelName='vgg11')
        visualModel = visualModel.to(device)
        state = torch.load(visualModelPath)
        # state_dict=state['state_dict'] # just used for debug
        visualModel.load_state_dict(state['state_dict'])
        visualModel.eval()


    # print(visualModel_indexes)

    ensemble_PROBS=[]
    visual_PROBS=[]
    c3d_PROBS=[]
    resample_PROBS=[]
    with torch.no_grad():
        for inputs in test_dataloader:
            inputs_C3D=inputs.clone()
            inputs_RNN=inputs.clone()
            assert id(inputs_C3D)!=id(inputs_RNN)

            inputs_C3D=normalize_C3D(inputs_C3D).to(device)
            inputs_RNN=normalize_RNN(inputs_RNN).to(device)
            print(inputs_RNN.shape)
            assert id(inputs_C3D)!=id(inputs_RNN)

            ensemble_probs = torch.zeros((inputs.size(0), params.num_classes)).to(device)
            visual_probs = torch.zeros((inputs.size(0), params.num_classes)).to(device)
            c3d_probs = torch.zeros((inputs.size(0), params.num_classes)).to(device)
            resmaple_probs = torch.zeros((inputs.size(0), params.num_classes)).to(device)




            for i,model in enumerate(ensemble_models):
                outputs=model(inputs_C3D)
                probs=outputs.softmax(1)
                if i==c3d_index:
                    c3d_probs+=probs
                if i==resample_index:
                    resmaple_probs+=probs

                ensemble_probs+=probs# Just take the average of all predictions obtained by different models

            if visualModel:
                outputs = visualModel(inputs_RNN)
                outputs=outputs[:,-1,:]
                visual_probs=outputs.softmax(1)


            del inputs_RNN,inputs_C3D

            # probs/=len(models)
            c3d_PROBS.append(c3d_probs.detach().cpu())
            resample_PROBS.append(resmaple_probs.detach().cpu())
            ensemble_PROBS.append(ensemble_probs.detach().cpu())
            visual_PROBS.append(visual_probs.detach().cpu())

    ensemble_PROBS=torch.cat(ensemble_PROBS).numpy()
    c3d_PROBS=torch.cat(c3d_PROBS).numpy()
    resample_PROBS=torch.cat(resample_PROBS).numpy()
    visual_PROBS=torch.cat(visual_PROBS).numpy()

    ensemble_PREDS=np.argmax(ensemble_PROBS,1)
    c3d_PREDS=np.argmax(c3d_PROBS,1)
    resample_PREDS=np.argmax(resample_PROBS,1)
    visual_PREDS=np.argmax(visual_PROBS,1)

    data_paths=test_dataloader.dataset.data_files
    data_labels=test_dataloader.dataset.data_labels
    # print(PROBS)
    # print(PROBS.shape)
    ensemble_acc = np.sum(ensemble_PREDS == data_labels) / len(test_dataset)
    c3d_acc = np.sum(c3d_PREDS == data_labels) / len(test_dataset)
    resample_acc = np.sum(resample_PREDS == data_labels) / len(test_dataset)
    visual_acc = np.sum(visual_PREDS == data_labels) / len(test_dataset)

    ensemble_ba = metrics.balanced_accuracy_score(data_labels, ensemble_PREDS)
    c3d_ba = metrics.balanced_accuracy_score(data_labels, c3d_PREDS)
    resample_ba = metrics.balanced_accuracy_score(data_labels, resample_PREDS)
    visual_ba = metrics.balanced_accuracy_score(data_labels, visual_PREDS)


    print(ensemble_acc,ensemble_ba)
    print(c3d_acc,c3d_ba)
    print(resample_acc,resample_ba)
    print(visual_acc,visual_ba)

    n_times=67
    value = [[n_times, ensemble_acc, ensemble_ba, c3d_acc, c3d_ba, resample_acc, resample_ba, visual_acc, visual_ba]]
    print(value)
    if not os.path.isfile('all_metrics.csv') or os.stat('all_metrics.csv').st_size == 0:
        # 如果文件不存在或者是空文件，添加列标题
        header = ['index', 'ensemble_accuracy', 'ensemble_balanced accuracy', 'c3d_accuracy', 'c3d_balanced accuracy',
                  'resample_accuracy', 'resample_balanced accuracy', 'visual_accuracy', 'visual_balanced accuracy']
        with open('all_metrics.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    # 追加新数据
    with open('all_metrics.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for data in value:
            writer.writerow(data)



if __name__=='__main__':
    main()











