#zmq
import os
import numpy as np
from pathlib import Path
import cv2
import random

from utils import set_seed


from utils import set_seed

class FileProvider:
    def __init__(self,video_path:str,seq_length:int,resize_width_height:list,label:int,
                 total_num_calls:int=1,mode:str='train',seed:int=0):
        self.video_path=video_path# the path of video
        self.img_paths=sorted(list(Path(self.video_path).glob('*.jpg')))# list of sorted pathnames of all frames in the video
        self.seq_length=seq_length# the specified number of consecutive frames to be extracted each time
        self.resize_width_height=resize_width_height# [width,height]
        self.label=label #0/1/2/3
        self.num_samples=len(self.img_paths)# the number of images in this video
        self.mode=mode
        self.seed=seed

        max_num_calls=np.floor(self.num_samples / self.seq_length).astype(int)# the maximal number of times this video can be extracted
        self.total_num_calls=total_num_calls if max_num_calls>total_num_calls else max_num_calls# the specified number of times this video can be extracted

        self.num_calls=0
        self.extracted_index=[]# the starting index of the first extractions. Every video can be extracted up to twice

        # set_seed(manul_seed)

    def _get_num_samples(self):
        return len(os.listdir(self.video_path))

    def reset(self):
        '''prepare for extraction during next epoch'''
        self.num_calls=0
        self.extracted_index = []

    def total_num_calls_reached(self):
        return self.num_calls==self.total_num_calls

    def get_seqs(self):
        '''read consecutive `self.seq_length` images from the video
        Returns:
            buffer(numpy.array): store consecutive `self.seq_length` images. Its shape is (self.seq_length,self.resize_width_height[1],self.resize_width_height[0],3)
            label(int): the label of this video
        '''
        if self.mode=='test':
            seed=111+self.seed
            set_seed(seed)
        buffer=np.empty((self.seq_length,self.resize_width_height[1],self.resize_width_height[0],3),np.dtype('float32'))# saves consecutive `self.seq_length` frames
        if self.total_num_calls==1:
            # this video will be extracted only once
            start_index=np.random.randint(self.num_samples-self.seq_length+1)
            end_index=start_index+self.seq_length
        else:
            # this video will be extracted twice
            if self.num_calls==0:
                # the first extraction: randomly extract consecutive `self.seq_length` frames
                start_index=np.random.randint(self.num_samples-self.seq_length*2+1)
                end_index=start_index+self.seq_length
                self.extracted_index.append(start_index)
            else:
                if self.extracted_index[0]>=self.seq_length:
                    # the number of frames whose indexes are less than the start index of the first extraction is no less than `self.seq_length`.
                    if round(random.random())==0:
                        # the second extracted images are from those indexes are less than the start index of the first extraction
                        start_index=np.random.randint(self.extracted_index[0]-self.seq_length+1)
                        end_index=start_index+self.seq_length
                    else:
                        # extracts from those indexes are more than the end index of the first extraction
                        start_index=np.random.randint(self.extracted_index[0]+self.seq_length,self.num_samples-self.seq_length+1)
                        end_index=start_index+self.seq_length
                else:
                    start_index = np.random.randint(self.extracted_index[0] + self.seq_length,
                                                    self.num_samples - self.seq_length+1)
                    end_index = start_index + self.seq_length
        self.num_calls+=1
        for i in range(start_index,end_index):
            frame=np.array(cv2.imread(str(self.img_paths[i]))).astype(np.float64)
            # print(frame.shape)
            assert frame.shape==(self.resize_width_height[1],self.resize_width_height[0],3)
            buffer[i-start_index]=frame
        assert buffer.shape==(self.seq_length,self.resize_width_height[1],self.resize_width_height[0],3)
        return buffer,self.label

if __name__=="__main__":
    t=np.array([[1,2],[1,2]])
    print(t.shape)





