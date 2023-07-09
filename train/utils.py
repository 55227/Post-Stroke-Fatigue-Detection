#zmq
import argparse
import os
import glob
import torch
import numpy as np
import random
import json

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))# the directory where this file is located

def parse_args():
    '''In the begining, I want to use command-line parameters to specify those parameters used for this project. But
    there are some dictionary-type parameters that cann't be specified by command-line parameters. So I decide to abandon this '''
    parser=argparse.ArgumentParser()
    parser.add_argument('--n-epochs',dest='num_epochs',type=int,default=20,help="Number of epochs for training")
    parser.add_argument('--resume-epoch',type=int,default=0,help="The epoch from which to resume")
    parser.add_argument('--useTest',type=bool,default=True,help="Whether testing is required")
    parser.add_argument('--test_interval',type=int,default=5,help="Run on test set every nTestInterval epochs")
    parser.add_argument('--snapshot',dest='save_epoch',type=int,default=10,help="Store a model every snapshot epochsg")
    parser.add_argument('--lr',type=float,default=1e-4,help="Learning rate")
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--modelName',type=str,default='C3D',choices=['C3D','visualRNNModel'])# C3D/visualRNNModel
    parser.add_argument('--visualModelName',type=str,default=None,choices=['resnet18','resnet34','resnet50','resnet101','vgg11'],help='the model name of cnn model used by visualRNNModel')
    parser.add_argument('--pretrained',type=str,default=True,help='decide whether to use pretrained model or train from scratch')
    parser.add_argument('--save_dir_root',type=str,default=save_dir_root,help="the directory where the `utils.py` file is located")
    parser.add_argument('--num_classes',type=int,default=4,help="Number of classes to be classified")
    parser.add_argument('--dataset',type=str,default='stroke_dataset')
    parser.add_argument('--seed',type=int,default=111)



    args, _ = parser.parse_known_args()# this method is uesd to partially parse. It doesn't produce an error when some extra arguments present in command line.
    return args

def resume_ckpt(model,resume_epoch,modelName,visualModelName):
    if modelName=='C3D':
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run','ckpt',modelName,'run_*')))
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run','ckpt',modelName,visualModelName,'run_*')))
    if resume_epoch != 0:  # resume last interrupted training. This parameter is the epoch before the interrupted epoch
        run_id = int(runs[-1].split('_')[-1]) if runs else 0  # the index of last interrupted training
        save_dir = os.path.join(save_dir_root,'run','ckpt',modelName,  'run_' + str(run_id))  # the directory where the files storing the model parameters during last training are located
        checkpoint = torch.load(
            os.path.join(save_dir,  'epoch-' + str(resume_epoch - 1) + '.pth.tar'),map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else: # training from scratch
        # runs = sorted(glob.glob(os.path.join(save_dir_root, 'run','ckpt',modelName, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    return model,run_id

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

class AverageMeter():
    '''Computes and stores the average and current value. Inspired by https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/imagenet/main.py#L420'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count


class Params:
    """ Helper class for the hyper-parameters.
        It can either take a dictionary or read from json file.
    """

    def __init__(self,
                 dict_params: dict = None,
                 json_path: str = None):

        if json_path:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        else:
            self.__dict__.update(dict_params)

    def save_to_json(self, json_path: str):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path: str):
        """Load parameters from json file and update dictionary."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __getitem__(self, key: str):
        return self.__dict__[key]

    def __str__(self):
        return str(self.dict)

    @property
    def dict(self):
        return self.__dict__  # containes all attributes and methods of class

def save_dict_to_json(dictionary: dict,
                      json_path: str):
    """ Saves dict of floats in json file

    Args:
        dictionary (dict): of float-castable values (np.float, int, float, etc.)
        json_path (string): path to json file
    """

    with open(json_path, 'a') as f:
        json.dump(dictionary, f, indent=4)

if __name__=='__main__':
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'ckpt', 'C3D', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    print(run_id)








