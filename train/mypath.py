import os

from utils import save_dir_root

class MyPath(object):
    @staticmethod
    def db_dir():
        # if database == 'stroke_dataset':
            # folder that contains class labels
        root_dir = r"D:\softdata\dataset\stroke_dataset"

        output_dir = os.path.join(save_dir_root,'stroke_dataset')

        return root_dir, output_dir



    @staticmethod
    def model_dir():
        return r"D:\BaiduNetdiskDownload\ucf101-caffe.pth" #预训练模型所在路径