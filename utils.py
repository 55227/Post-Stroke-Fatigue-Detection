#zmq
import torch
import os
import cv2

from network import model_provider

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))# the directory where this file is located


def load_model(path,device):
    try:
        model = model_provider.get_model().to(device)
        state = torch.load(path)
        model.load_state_dict(state['state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(repr(e))


class Camera:
    '''get the number of cameras in this computer and the indices of available cameras'''
    def __init__(self, cam_preset_num=5):
        self.cam_preset_num = cam_preset_num

    def get_cam_num(self):
        cnt = 0
        devices = []
        for device in range(0, self.cam_preset_num):
            stream = cv2.VideoCapture(device, cv2.CAP_DSHOW)
            grabbed = stream.grab()
            stream.release()
            if not grabbed:
                continue
            else:
                cnt = cnt + 1
                devices.append(device)
        return cnt, devices