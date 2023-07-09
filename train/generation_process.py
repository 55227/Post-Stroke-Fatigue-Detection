#zmq
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing
import cv2
import os
from pathlib import Path


from base_process import BaseProcess
from utils import save_dir_root
from mypath import MyPath


class GenerationProcess(BaseProcess):
    def __init__(self,resize_height:int=128,resize_width:int=171,seq_length:int=16,sample_strategy:dict=None):
        '''
         Args:
            dataset (str): Name of dataset. Defaults to 'stroke_dataset'.
            seq_length (int): Determines the number of consecutive frames one sample should contain. Defaults to 16.
            sample_strategy (dict): Contains the information to sample the dataset (only for train split). If `None`, no resampling; if a `dict`,
                                    The keys correspond to the targeted class from which to sample and the value are the number of samples to resample.

        '''
        self.root_dir, self.output_dir = MyPath.db_dir()  # the path of original dataset and the path to store the processed dataset
        self.resize_height=resize_height
        self.resize_width=resize_width
        self.seq_length=seq_length
        self.sample_strategy=sample_strategy

        logger_dir=Path(save_dir_root)
        self.set_logger(str(logger_dir/'generation.log'))
        logging.info('Starting Generation Process')

        if (self.sample_strategy is not None) and (not isinstance(self.sample_strategy, dict)):
            raise ValueError("'sample_strategy' must be None or dict type, but recieved is {0} type".format(
                type(self.sample_strategy)))

        if not self.check_integrity():  # if `self.root_dir` doesn't exist, raise an error
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

    def check_integrity(self):
        '''check if `self.root_dir` exists'''
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def process_video(self, video, action_name, save_dir):
        '''
        read `video` and capture one every `EXTRACT_FREQUENCY` frames (the initial value of `EXTRACT_FREQUENCY` is 4 and
        its value will be decreased when there are not enough frames to extract `self.seq_length` frames) and save the captured frames
        to the folder under `save_dir`. Names the frames so that the length of their name is the same

        Args:
            video: filename of the video
            action_name: the class name
            save_dir: the pathname of `action_name` folder, such as `self.output_dir/train/action_name`
        '''

        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]  # the video filename without the extension `.wav`
        if not os.path.exists(os.path.join(save_dir, video_filename)):  # ensure the folder to store the captured frames exists
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))  # read the video

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # the total number of frames
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of the frame
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the height
        num_digits=len(str(frame_count))# the number of digits in `frame_count`

        # Make sure splited video has at least `self.seq_length` frames
        EXTRACT_FREQUENCY = 4  ##capture one every `EXTRACT_FREQUENCY` frames
        if frame_count // EXTRACT_FREQUENCY <= self.seq_length:  # when `frame_count` is no more than `self.seq_length*4`，substract 1 from the `EXTRACT_FREQUENCY` value
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= self.seq_length:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= self.seq_length:
                    EXTRACT_FREQUENCY -= 1

        count = 0  # indicates which frame is being read
        i = 0  # indicates which frame is being captured. Use it to name the frame
        retaining = True

        while (count < frame_count and retaining):  # 当视频帧数没有被读取完
            retaining, frame = capture.read()  # 读取一帧，retaining说明是否读取到帧，若无，则为False，frame返回读取到帧的图像
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):  # 帧尺寸不满足128*171时进行resize
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                    assert np.array(frame).shape==(self.resize_height,self.resize_width,3)
                imgname='{:0{}}.jpg'.format(i,num_digits)
                assert len(imgname)==(num_digits+4)
                cv2.imwrite(filename=os.path.join(save_dir, video_filename,imgname ),
                            img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def start(self):
        # Given the the `output_dir` path doesn't exist, create it and create 'train', 'valid' and 'test' folder under this folder
        if not os.path.exists(self.output_dir):  # the `output_dir` path doesn't exist
            os.mkdir(self.output_dir)  # create `self.output_dir` directory
            os.mkdir(os.path.join(self.output_dir, 'train'))  # create `train` directory in `output_dir` directory
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):  # `file` is one of classes
            file_path = os.path.join(self.root_dir, file)  # the path of the folder storing all videos belonging to `file` class
            video_files = [name for name in os.listdir(file_path)]  # list of filenames of videos belonging to `file` class
            ####split the dataset.The proportion of training set is 0.64, the proportion of validation set is 0.16, and the proportion of test is 0.2
            train_and_valid, test = train_test_split(video_files, test_size=0.2,
                                                     random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)# `train` is list of filenames of videos belonging to training set

            train_dir = os.path.join(self.output_dir, 'train', file)  # the path of `file` folder
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
            #### over-sampling or down-sampling the train split
            if self.sample_strategy is not None:
                if file in self.sample_strategy.keys():

                    random_state = np.random.RandomState(16)
                    train_len = len(train)
                    target_class_indices = [i for i in range(train_len)]

                    if self.sample_strategy[file] > 0:
                        # over-sampling
                        bootstrap_indices = random_state.choice(target_class_indices,
                                                                size=self.sample_strategy[file],
                                                                replace=False)  # `replace` parameter indicated whether the indexing can be repeated
                        train_bootstrap = _safe_indexing(train, bootstrap_indices)# list of filenames of some videos to be resampled
                        train = train + train_bootstrap
                    else:
                        # down-sampling
                        bootstrap_indices = random_state.choice(target_class_indices,
                                                                size=train_len + self.sample_strategy[file],
                                                                replace=False)  # `size` parameter must be calculated using + instead of -, because `self.sample_strategy[file]` is negative
                        train = _safe_indexing(train, bootstrap_indices)# list of filenames of videos after removing those indexed by `bootstrap_indices`

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)


if __name__=="__main__":
    generation=GenerationProcess()
    generation.start()



