# zmq
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import torch
from torchvision import transforms


from utils import set_seed
from data_provider.file_provider import FileProvider
from mypath import MyPath


class TestProvider(Dataset):
    def __init__(
            self,
            modelName: str = 'C3D',  # used to indicate which normalization operation to use
            resize_width_height: list=[171,128],
            crop_size:int=112,
            # manul_seed: int = 0, # actually according to the original `dataset.py`, there is no need to set the same random seed every time
            seq_length: int = 16,
            seed:int=67,
    ):
        '''
        Args:
            resize_wdith_height: the size of images
            split: train/val/test. Indicate this is used for training, validation or test
            crop_size: the size of cropped images
            seq_length: the number of consecutive frames to extract every time
            sampling_strategy: Only for train set. the keys are names of classes and the corresponding values is the number of samples to be resampled. If the value is negative, downsample that class

        '''
        _,root_dataset=MyPath.db_dir()
        self.dataset_path = os.path.join(root_dataset,'test')# the path of train/valid/test dataset
        self.seq_length = seq_length
        self.resize_width_height = resize_width_height
        self.split='test'
        self.modelName=modelName
        self.crop_size=crop_size
        self.classes = ['drowsiness', 'inattention', 'normal', 'yawn']
        self.class2index={self.classes[i]:i for i in range(len(self.classes))}# the key-value pairs are the class names and their corresponding numerical representations
        # the path of folder storing all images belonging to that class in
        # `dataset_path` directory
        self.class_dataset_path = {
            name: os.path.join(
                self.dataset_path,
                name) for name in self.classes}

        # the values correspond to the pathnames of all video folders belonging
        # to the class
        self.class_data_files = {
            name: sorted(
                list(
                    Path(
                        self.class_dataset_path[name]).glob('*'))) for name in self.classes}


        # the values are the number of video folders belonding to that class
        self.class_num_data_files = {
            name: len(self.class_data_files[name]) for name in self.classes}
        self.total_num_videos=sum(self.class_num_data_files.values())# the total number of videos in train/val/test set

        self.data_files = [file for class_files in self.class_data_files.values() for file in class_files]
        self.data_labels = [0]*self.class_num_data_files['drowsiness'] + [1]*self.class_num_data_files['inattention']+ [2]*self.class_num_data_files['normal'] + [3]*self.class_num_data_files['yawn']
        assert len(self.data_files)==len(self.data_labels)
        # set_seed(manul_seed)

        self.total_data_providers=[FileProvider(self.data_files[i],self.seq_length,resize_width_height,self.data_labels[i],1,'test',seed) for i in range(len(self.data_files)) ]


    def reset(self):
        '''Prepares for next epoch'''
        for i in self.total_data_providers:
            i.reset()


    def _get_data_provider(self,idx):
        i=idx%self.total_num_videos
        file=self.total_data_providers[i]
        while file.total_num_calls_reached():
            i=(i+1) if (i+1)<self.total_num_videos else 0
            file=self.total_data_providers[i]
        return file

    def center_crop(self,buffer):
        '''Perform center crop on image
        Args:
            buffer(numpy.array)
        Returns:
            buffer(torch.Tensor)
        '''
        buffer=buffer[:,8:120,30:142,:]
        buffer=torch.from_numpy(buffer)
        return buffer




    def process_input(self,data):
        '''Image augmentation: center crop (if this dataset is used for test)->normalize->transpose
           Args:
               data(numpy.array)
           Returns:
               data(torch.Tensor)

        '''
        data=self.center_crop(data)

        # if self.modelName=="C3D":
        #     data=self.normalize_C3D(data)
        #     data = self.dimension_conversion_C3D(data)
        # else:
        #     # data=self.normalize_RNN(data)
        #     data = self.dimension_conversion_RNN(data)
        #
        # data=torch.from_numpy(data)
        # if self.modelName=='VisualRNNModel':
        #     for i in range(data.shape[0]):
        #         data[i]=self.normalize_RNN(data[i])
        return data

    def __len__(self):
        return sum([x.total_num_calls for x in self.total_data_providers])

    def __getitem__(self, idx):
        '''
        Returns:
            data(torch.Tensor): Its shape is (seq_length, height, width, channels)
        '''
        data_provider=self._get_data_provider(idx)
        data,_=data_provider.get_seqs()
        data=self.process_input(data)
        return data

def normalize_C3D(buffer):
    '''For a three-channel image, subtract a specified value from each channel's pixel value
    Args:
        buffer(torch.tensor): its shape is (batch_size,seq_length, height, width, channels)
    Returns:
        buffer(torch.tensor): its shape is (batch_size,channels, seq_length, height, width)
    '''
    for i,batch in enumerate(buffer):
        for j,frame in enumerate(batch):
            try:
                frame -= torch.tensor([[[90.0, 98.0, 102.0]]]) #Three-channel image:
                buffer[i,j] = frame
            except:
                print(buffer.shape)
                raise ValueError(buffer.shape)
    return torch.permute(buffer,(0,4,1,2,3))

def normalize_RNN(buffer):
    '''
    Args:
        buffer(torch.tensor): (batch_size,seq_length, height, width, channels)
    Returns:
        buffer(torch.tensor): ( batch_size,seq_length, channels,height, width)
    '''
    buffer=torch.permute(buffer,(0,1,4,2,3))# ( batch_size,seq_length, channels,height, width)
    for i,batch in enumerate(buffer):
        for j,frame in enumerate(batch):
            frame=normalize(frame)
            buffer[i,j] = frame
    return buffer

def normalize(buffer):
    '''for CNN+RNN model, it needs to normalize data
    Args:
        buffer(torch.tensor)
    '''
    normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    return normalize(buffer/255.)



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    test_dataloader=DataLoader(TestProvider(),batch_size=5)
    for data in test_dataloader:

        if test_dataloader.dataset.modelName=='C3D':
            data=normalize_C3D(data)
        else:
            data=normalize_RNN(data)
        # print(type(data))
        # print(data.shape)




