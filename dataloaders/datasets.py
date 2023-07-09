#zmq
from pathlib import Path
import cv2
import os
import numpy as np

class LoadFrames:
    def __init__(self,path,resize_width_height:tuple=(171,128),crop_size:int=112):
        print('数据集开始初始化')
        p=str(Path(path).resolve())# os-agnostic absolute path
        if os.path.isfile(p):
            self.file=p
        else:
            raise Exception(f'错误：{p} 不存在！')
        self.resize_width_height = resize_width_height
        self.crop_size=crop_size
        self.new_video(self.file)
        print('数据集初始化成功')

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame==self.frames:
            raise StopIteration

        ret_val, frame= self.cap.read()  # `img` is a grabbed frame in this video file
        data=self.center_crop(cv2.resize(frame,self.resize_width_height))
        data=data-np.array([[[90.0, 98.0, 102.0]]])
        data=np.ascontiguousarray(data)
        self.frame+=1

        return data,frame

    def new_video(self, path):
        self.frame = 0  # # the number of frames already read
        self.cap = cv2.VideoCapture(
            path)  # a VideoCapture object to read frames from a video file specified by `path`
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # the total number of frames in the video

    def center_crop(self,frame):
        '''center crop
        Args:
            frame: its shape is (height,width,channels)
        '''
        frame = frame[8:120, 30:142, :]  # 根据原始尺寸128*171得到重新裁剪的像素范围
        return np.array(frame).astype(np.uint8)

class LoadCam:
    def __init__(self,pipe='0',resize_width_height:tuple=(171,128),crop_size:int=112):
        self.resize_width_height = resize_width_height
        self.crop_size = crop_size
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe# `eval` used to convert `pipe` to int type. The original type of `pipe` is numeric string
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size
        self.frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))# prepare for progressBar
        print('摄像头数据集初始化完成')

    def iter(self):
        return self

    def __next__(self):
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration
        # Read frame
        ret_val, frame = self.cap.read()
        frame=cv2.flip(frame,1)
        assert ret_val, f'Camera Error {self.pipe}'
        data=self.center_crop(cv2.resize(frame,self.resize_width_height))
        data=data-np.array([[[90.0, 98.0, 102.0]]])
        data=np.ascontiguousarray(data)

        return data,frame

    def center_crop(self,frame):
        '''center crop
        Args:
            frame: its shape is (height,width,channels)
        '''
        frame = frame[8:120, 30:142, :]  # 根据原始尺寸128*171得到重新裁剪的像素范围
        return np.array(frame).astype(np.uint8)








