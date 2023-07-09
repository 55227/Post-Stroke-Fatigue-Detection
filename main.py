#zmq

import sys
import time
import json
import numpy as np
import os
from pathlib import Path
import cv2
import requests
from playsound import playsound


import torch

from PyQt5.QtCore import  Qt, QPoint, QTimer,QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5.QtWidgets import *
from PyQt5 import uic,QtWidgets

from login import Ui_Form
from win import Ui_mainWindow
from network.model_provider import get_model
from dataloaders.datasets import LoadFrames,LoadCam
from utils import load_model,Camera
from CustomMessageBox import MessageBox

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dir = r'C:\Users\dell\Desktop\graduate_codeLib\graduate_project_code\post_stroke_fatigue_detection\run\ckpt\C3D'# TODO
open_fold=r'D:\softdata\dataset\stroke_dataset' # the directory to select a video



class DetThread(QThread):
    send_img=pyqtSignal(np.ndarray)
    send_raw=pyqtSignal(np.ndarray)
    send_prediction_label=pyqtSignal(dict)

    send_msg=pyqtSignal(str)
    send_percent=pyqtSignal(int)
    send_intervene=pyqtSignal()

    def __init__(self):
        super(DetThread, self).__init__()
        self.weight=None# str: the path of a selected trained model file or `集成模型` string which means it will use ensemble model
        self.current_weight=None# the path of a trained model file in use or `ensemble` string
        # self.video_file=None # the path of a selected video file to be inferenced
        self.jump_out=False # To indicate whether to stop this thread
        self.is_continue=True # To determine whether to continue the detection or pause it
        self.percent_length=1000 # the length of the progress bar widget
        self.source=None # the pathname of a selected file to be detected or the index of selected camera
        self.index2class=['drowsiness','inattention','normal','yawn']
        # self.index2class=['昏睡','注意力不集中','正常','打哈欠']

        self.play_once=False # indicate whether the audio has been played once
        # self.ensemble_model_list=None

    @torch.no_grad()
    def run(self):

        print('DetThread线程运行')
        print('选择文件：',self.source)
        print(self.weight)
        if self.weight!='集成模型':
            model=load_model(self.weight,device)
        else:
            # ensemble all models whose accuracy is highest
            model_list=[]
            model_file_list = list(Path(model_dir).glob("**/*.pth.tar"))  # list of pathnames of model files
            for file in model_file_list:
                model=load_model(file,device)
                model_list.append(model)


            pass

        if self.source.isnumeric():
            dataset=LoadCam(self.source)
        else:
            dataset=LoadFrames(self.source)

            label=str(self.source).split('/')[-2]# the label of the selected video: cann't use `os.sep` as the seperator. In windows, it's '\', but the seperator in `self.source` is '/'
            print(label)
        frame_count=0 # the number of read frames
        frame_drowsiness=0 # the number of frames predicted as drowsiness
        clip=[]
        print(self.jump_out,'jump out')

        while True:
            if self.jump_out:
                try:
                    print('结束此次检测')
                    self.play_once = False
                    self.send_percent.emit(0)
                    self.send_msg.emit('结束此次检测')
                    break
                except Exception as e:
                    print(repr(e))
                    break
            if self.current_weight!=self.weight:
                # In case the selected model changes
                if self.weight != '集成模型':
                    model = load_model(self.weight, device)
                else:
                    model_list = []
                    model_file_list = list(Path(model_dir).glob("**/*.pth.tar"))  # list of pathnames of model files
                    for file in model_file_list:
                        model = load_model(file, device)
                        model_list.append(model)
                self.current_weight=self.weight
            if self.is_continue:
                print('is continue')
                data,frame=next(dataset)
                clip.append(data)
                frame_count+=1

                percent=int(frame_count/dataset.frames*self.percent_length)
                self.send_percent.emit(percent)
                if len(clip)==16:
                    try:
                        inputs=np.array(clip).astype(np.float32)#(16,112,112,3) # 无错误
                        inputs = np.expand_dims(inputs, axis=0)  # (1,16,112,112,3)#满足conv3D的输入shape要求
                        inputs = np.transpose(inputs, (0, 4, 1, 2, 3))  # (1,3,16,112,112)
                        inputs = torch.from_numpy(inputs)
                        inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)

                        if self.weight!='集成模型':
                            outputs = model(inputs)
                            probs = torch.nn.Softmax(dim=1)(outputs)

                            pass
                        else:
                            print('模型集成')
                            probs=torch.zeros((1,4)).to(device)
                            for model in model_list:
                                outputs=model(inputs)
                                probs+=outputs.softmax(1)
                                del outputs
                            probs/=len(model_list)
                            print(f"集成{len(model_list)}个模型")

                        clip.pop(0)

                        prediction_index = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
                        prediction_class=self.index2class[prediction_index]
                        English2Chinese={'drowsiness':'昏睡','inattention':'注意力不集中','normal':'正常','yawn':'打哈欠'}
                        if self.source.isnumeric():
                            prediction_label_dict={'预测疲劳程度':English2Chinese[prediction_class]}
                        else:
                            prediction_label_dict={'预测疲劳程度':English2Chinese[prediction_class],'标签':English2Chinese[label]}

                        if (prediction_index==0) and (not self.play_once):
                            frame_drowsiness+=1
                            # drowsiness is detected
                            if frame_drowsiness>=10:
                                self.play_once=True
                                print('emit the intervene signal')
                                self.send_intervene.emit()

                        # print(img.shape)
                        frame_out=frame.copy()
                        cv2.putText(frame_out, prediction_class, (80, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 255), 2)  # 在frame图像上渲染文字（第二个参数），（20,20）表示文字的位置，0.6表示字体大小，红色字体，1.5表示线的粗细
                        cv2.putText(frame_out, "prob: %.4f" % probs[0][prediction_index], (80, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 255), 2)

                        assert id(frame_out)!=id(frame)
                        self.send_raw.emit(frame)
                        self.send_img.emit(frame_out)
                        self.send_prediction_label.emit(prediction_label_dict)

                        if percent==self.percent_length:
                            self.send_percent.emit(0)
                            self.send_msg.emit('检测完成')
                            break

                    except Exception as e:
                        print('detthread出错')
                        print(repr(e))


class DetectWin(QMainWindow,Ui_mainWindow):
    def __init__(self,parent=None):
        print('MainWindow初始化')
        super(DetectWin, self).__init__(parent)
        self.setupUi(self)

        # style 1: window can be stretched
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # # style 2: window can not be stretched
        # self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
        #                     | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)  # More: https://blog.csdn.net/lion_cxq/article/details/117306972
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        # self.maxButton.animateClick(10)  # the buttoon is pressed immediately, and released 10 milliseconds later
        self.closeButton.clicked.connect(self.close)

        # self.qtimer = QTimer(self)
        # self.qtimer.setSingleShot(True)  # set this timer is a single-shot timer, i.e., this timer fires only once
        # self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        self.comboBox.clear()  # the widgets used to select model
        self.pt_list = list(Path(model_dir).glob("**/*.pth.tar"))# list of pathnames of models
        self.pt_list.sort(key=lambda x: os.path.getsize(x))  # `x` is the pathname. This expression is used to sort these files by file size as sorting criteria
        # self.pt_list=[os.path.basename(pt) for pt in self.pt_list]# list of filenames of models
        self.pt_list=[str(pt) for pt in self.pt_list]
        self.pt_list.append('集成模型')
        self.basename_list=[os.path.basename(pt).split('.')[0] for pt in self.pt_list]# the filename of all models
        # self.basename_list.append('集成模型')
        self.comboBox.clear()
        self.comboBox.addItems(self.basename_list)
        # self.comboBox.setCurrentIndex(0)# set the first item as the default item

        # set the timer which will keep counting until it reaches the specified time interval, and then will restart counting from 0
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        self.det_thread=DetThread()
        self.det_thread.weight=[x for x in self.pt_list if self.comboBox.currentText() in x][0]
        self.det_thread.percent_length=self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x:self.show_image(x,self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_prediction_label.connect(self.show_prediction_label)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_intervene.connect(self.intervene)

        self.fileButton.clicked.connect(self.open_file)# select a video or a image
        self.cameraButton.clicked.connect(self.chose_cam)

        self.runButton.clicked.connect(self.run_or_continue)  # if `runButton` is clicked, the `self.det_thread.jump_out` variable will be set to False
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)

        self.saveCheckBox.setCheckState(0)# set `疲劳干预` widget unchecked initially


    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    @staticmethod
    def show_image(img,widget1):
        try:
            ih,iw,_=img.shape# TODO
            w=widget1.geometry().width()
            h=widget1.geometry().height()
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_ = cv2.resize(img, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_ = cv2.resize(img, (nw, nh))
            frame=img_
            frame=cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
            img=QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            widget1.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_prediction_label(self,prediction_label_dic):
        # display the prediction and gound truth
        # TODO
        try:
            self.resultWidget.clear()
            results = [' '+str(i) + '：' + str(j) for i,j in prediction_label_dic.items()]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def show_msg(self, msg):
        # TODO
        self.runButton.setChecked(Qt.Unchecked)# Set the widget unchecked. `Qt.Unchecked` represent the unchecked state.
        self.statistic_msg(msg)
        if msg == "检测完成":
            self.saveCheckBox.setEnabled(True)
        elif msg=='结束此次检测':
            self.resultWidget.clear()# clear the prediction and ground truth information
            self.raw_video.clear()# clear the original image
            self.out_video.clear()# clear the image

    def intervene(self):
        print('intervene function')
        if self.saveCheckBox.isChecked():
            # To play the audio, I first try to use `os.system`, but it must open a software. It isn't convenient. I
            # then try to use `playsound` library, but it always goes wrong. At first I thought it was because the audio
            # path contains Chinese characters. But when I run this code block separately, it's okay. I still
            # convert the path to the one which doesn't contain Chinese characters. But sometimes it also goes wrong.
            # Finally, I modify its source code, More refers to https://blog.csdn.net/qq_43539004/article/details/120515604

            hint=r'C:\Users\dell\Desktop\graduate_codeLib\ui界面\me\audio\drowsiness.mp3'
            playsound(hint)
            self.statistic_msg('检测到患者处于疲劳状态！')

    def statistic_msg(self, msg):
        '''display text information on widget'''
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)

    def search_pt(self):
        pt_list=list(Path(model_dir).glob("**/*.pth.tar"))
        pt_list.sort(key=lambda x: os.path.getsize(x))
        pt_list=[str(pt) for pt in pt_list]
        pt_list.append('集成模型')
        basename_list=[os.path.basename(pt).split('.') for pt in pt_list]
        # basename_list.append('集成模型')
        # pt_list=[os.path.basename(pt) for pt in pt_list]# list of filenames of models
        if pt_list!=self.pt_list:
            self.pt_list=pt_list
            self.basename_list=basename_list
            self.comboBox.clear()
            self.comboBox.addItems(self.basename_list)
            # self.comboBox.setCurrentIndex(0)

    def open_file(self):

        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")  # `name is the path of selected file
        if name:
            self.det_thread.source = name
            self.statistic_msg('选择文件：{}'.format(os.path.basename(name)))  # the filename of selected file

            self.stop()

    def chose_cam(self):
        try:
            self.stop()
            # display a message box whose title is `title` and text is `text`. And it will close automatically after 2 seconds
            MessageBox(# a custom message box
                self.closeButton, title='提示', text='正在加载摄像头', time=2000, auto=True).exec_()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()# `cams` is a list of indexes of availabel cameras
            popMenu = QMenu()# menu bar
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:# traverse the available camera device indexed, create corresponding QAction, and add them to the menu bar
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            # set the popup position of the menu below `cameraButton` widget. Uses `exec_` function to display the menu and waits for the user to select a menu item
            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()# map the `cameraButton` widget coordinate to global screen coordinates
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()# the index of selected camera
                self.statistic_msg('加载摄像头：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)


    def run_or_continue(self):
        # TODO
        self.det_thread.jump_out = False
        if self.runButton.isChecked():  # if the `runButton` is checked
            self.runButton.setToolTip('暂停')
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True  # continue to detect
            if not self.det_thread.isRunning():  # if `self.det_thread` thread is not running
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('检测中 >> 模型：{}，视频文件：{}'.
                               format(os.path.basename(self.det_thread.weight),
                                      source))
        else:
            self.runButton.setToolTip('开始')
            self.det_thread.is_continue = False # pause the detection
            self.statistic_msg('暂停')

    def stop(self):
        # self.runButton.setCheckState(0)
        self.runButton.setToolTip('开始')
        self.resultWidget.clear()
        self.raw_video.clear()
        self.out_video.clear()
        self.det_thread.jump_out=True
        self.saveCheckBox.setEnabled(True)

    def change_model(self):
        # TODO
        try:
            # items=[self.comboBox.itemText(i) for i in range(self.comboBox.count())]
            # print(items)
            # print(len(items))
            self.det_thread.weight =[x for x in self.pt_list if self.comboBox.currentText() in x][0]
            # self.det_thread.weight =self.comboBox.currentText()
            self.statistic_msg('将模型变更为：%s' % self.comboBox.currentText())
            print(self.comboBox.currentText())
        except Exception as e:
            print(repr(e))




class LoginThread(QThread):
    '''the sub-thread for login'''
    start_login_signal = pyqtSignal(str)  # `str` indicates the slot function binded to this signal receive the parameter of type string
    def __init__(self,signal):
        super(LoginThread, self).__init__()
        self.login_complete_signal=signal
        self.jump_out=False

    def login_by_requests(self, user_password_json):
        user_password_json = json.loads(user_password_json)
        print(user_password_json.get('username'))
        print(user_password_json.get('password'))

        #  uses the request module to send the request
        r=requests.post(url="https://service-1h60wox7-1317024607.sh.apigw.tencentcs.com/release/qt_login",json=user_password_json)
        print('receive the response from the tencent cloud function',r.content.decode())
        ret=r.json()

        print('send the message to main thread')
        self.login_complete_signal.emit(json.dumps(ret))

    def run(self):
        while True:  # keep the sub-thread running continuously so that it can receive signals from the main thread
            if self.jump_out:
                break
            print('sub-thread is running')
            time.sleep(1)


class LoginUi(QtWidgets.QWidget,Ui_Form):
    # construct a custom signal
    login_status_signal=pyqtSignal(str)

    def __init__(self):
        super(LoginUi, self).__init__()
        self.setupUi(self)
        self.init()

    def init(self):

        self.pushButton.clicked.connect(self.login)
        # connect the custom signal to `login_status` function so that the sub-thread can send information to main thread
        self.login_status_signal.connect(self.login_status)
        # create a child thread
        self.login_thread=LoginThread(self.login_status_signal)
        self.login_thread.start_login_signal.connect(self.login_thread.login_by_requests)
        # start the child thread
        self.login_thread.start()


    def login(self):
        '''implement login'''
        username=self.lineEdit_4.text()
        password=self.lineEdit.text()
        print(username,password)
        print('=====')

        self.login_thread.start_login_signal.emit(json.dumps({"username":username,"password":password}))

    def login_status(self,status):
        status=json.loads(status)
        # print("status...",status)
        # print(type(status))
        # print(type(status['errno']))
        if status['errno']==0:# sucessfully login in
            print('successfully login in')
            self.login_thread.jump_out=True# To stop the `self.login_thread`
            self.detectMainWindow=DetectWin()# To prevent the new window from flashing, it's necessary to bind the new window to a member variable
            print('初始化完成')
            # self.close()
            self.detectMainWindow.show()
            self.close()
        elif status['errno']==1001:
            QtWidgets.QMessageBox.critical(self, "用户不存在", "请重新输入")
            self.lineEdit_4.clear()
            self.lineEdit.clear()
        else:
            QtWidgets.QMessageBox.critical(self, "密码错误", "请重新输入")
            self.lineEdit_4.clear()
            self.lineEdit.clear()


if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)

    w=LoginUi()
    w.show()
    sys.exit(app.exec_())
