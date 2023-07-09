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


class BaseProvider(Dataset):
    def __init__(
            self,
            split: str,
            resize_width_height: list=[171,128],
            modelName:str='C3D',# used to indicate which normalization operation to use
            crop_size:int=112,
            seq_length: int = 16,
            sampling_strategy: dict = None):
        '''
        Args:
            resize_wdith_height: the size of images
            split: train/val/test. Indicate this is used for training, validation or test
            crop_size: the size of cropped images
            seq_length: the number of consecutive frames to extract every time
            sampling_strategy: Only for train set. the keys are names of classes and the corresponding values is the number of samples to be resampled. If the value is negative, downsample that class

        '''
        _,root_dataset=MyPath.db_dir()
        self.dataset_path = os.path.join(root_dataset,split)# the path of train/valid/test dataset
        self.seq_length = seq_length
        self.resize_width_height = resize_width_height
        self.split=split
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

        self.resampled_class_num_files={k:v for k,v in self.class_num_data_files.items()}# used to assign weights to classes for calculating loss values
        self.sample_index= {name:[] for name in self.classes}

        if sampling_strategy:  # resampling. Only for training set
            self.sample_total_calls=dict()
            for name,sample_num in sampling_strategy.items():
                self.sample_index[name]=np.random.choice(np.arange(0,self.class_num_data_files[name]),size=abs(sampling_strategy[name]),replace=False)
                self.sample_total_calls[name]=2 if sample_num>0 else 0
                self.resampled_class_num_files[name]+=sample_num


        self.class_data_providers = {
            name: [
                (FileProvider(
                    self.class_data_files[name][i],
                    self.seq_length,
                    resize_width_height,
                    self.class2index[name],
                    self.sample_total_calls[name]) if i in self.sample_index[name] else FileProvider(
                    self.class_data_files[name][i],
                    self.seq_length,
                    resize_width_height,
                    self.class2index[name],
                ))for i in range(
                    self.class_num_data_files[name])] for name in self.classes}
        self.total_data_providers=[f for class_data_provider in self.class_data_providers.values() for f in class_data_provider]# list of the FileProvider object of all video folder in train/val/test set
        assert len(self.total_data_providers)==self.total_num_videos

    def reset(self):
        '''Prepares for next epoch'''
        for i in self.total_data_providers:
            i.reset()

    def loss_weights(self):
        '''assign different weights to different classes for calculating loss values'''
        weight_classes=[1/resampled_num_files for resampled_num_files in self.resampled_class_num_files.values()]
        total_weight=sum(weight_classes)
        weight_classes_final=torch.FloatTensor([weight/total_weight for weight in weight_classes])
        return weight_classes_final
        pass

    def _get_data_provider(self,idx):
        i=idx%self.total_num_videos
        file=self.total_data_providers[i]
        while file.total_num_calls_reached():
            i=(i+1) if (i+1)<self.total_num_videos else 0
            file=self.total_data_providers[i]
        return file

    def crop(self,buffer):
        # randomly crop the image
        height_index=np.random.randint(buffer.shape[1]-self.crop_size)
        width_index=np.random.randint(buffer.shape[2]-self.crop_size)
        buffer=buffer[:,height_index:height_index+self.crop_size,width_index:width_index+self.crop_size]
        return buffer

    def randomflip(self,buffer):
        if np.random.random()<0.5:
            for i,frame in enumerate(buffer):
                frame=cv2.flip(buffer[i],flipCode=1)# horizontally flip the given image with the probability 0.5
                buffer[i]=cv2.flip(frame,flipCode=1)
        return buffer

    def normalize_C3D(self,buffer):
        '''For a three-channel image, subtract a specified value from each channel's pixel value '''
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]]) #Three-channel image:
            buffer[i] = frame
        return buffer

    def normalize_RNN(self,buffer):
        '''for CNN+RNN model, it needs to normalize data '''
        normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        return normalize(buffer/255.)

    def dimension_conversion_C3D(self,buffer):
        '''transpose the dimensions of `buffer` from (seq_length, height, width, channels) to (channels, seq_length, height, width)
           This is done because the input size of 3D convolutional layer is specified as (batch_size,channels,seq_length,height,width)
        '''
        return buffer.transpose((3,0,1,2))

    def dimension_conversion_RNN(self, buffer):# TODO
        '''transpose the dimensions of `buffer` from (seq_length, height, width, channels) to ( seq_length, channels,height, width)
        '''
        return buffer.transpose((0,3,1,2))

    def process_input(self,data,label):
        '''Image augmentation: randomly crop->randomly flip (if this dataset is used for test)->normalize->transpose

           Returns:

        '''
        data=self.crop(data)
        label=np.array(label)# convert `label` from int type to numpy.array type

        if self.split=='test':
            data=self.randomflip(data)

        if self.modelName=="C3D":
            data=self.normalize_C3D(data)
            data = self.dimension_conversion_C3D(data)
        else:
            data = self.dimension_conversion_RNN(data)

        data=torch.from_numpy(data)
        label=torch.from_numpy(label)
        if self.modelName=='VisualRNNModel':
            for i in range(data.shape[0]):
                data[i]=self.normalize_RNN(data[i])

        return data,label


    def __len__(self):
        return sum([x.total_num_calls for x in self.total_data_providers])

    def __getitem__(self, idx):
        '''
        Returns:
            data(torch.Tensor): Its shape is (channels, seq_length, height, width)
            label(torch.Tensor):
        '''
        data_provider=self._get_data_provider(idx)
        data,label=data_provider.get_seqs()
        data,label=self.process_input(data,label)
        return data,label



if __name__ == "__main__":
    pro=BaseProvider(split='train',sampling_strategy={'normal':-10,'drowsiness':10})
    sample_index_normal=pro.sample_index['normal']
    print(pro.class_dataset_path)
    print(pro.total_num_videos)
    print(pro.sample_index)
    print('fileprovider:',pro.class_data_providers['normal'][sample_index_normal[0]].total_num_calls)
    print('number of samples:',len(pro))
    print('class_num_video:',pro.class_num_data_files)
    provider=pro.class_data_providers['normal'][sample_index_normal[0]]
    print(len(provider.img_paths))
    print(provider.img_paths)
    buffer,label=provider.get_seqs()
    print(buffer.shape)
    print(label)


