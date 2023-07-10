# 基于三维卷积神经网络的卒中患者疲劳检测
本项目构建基于迁移学习的C3D模型用于卒中后疲劳识别；针对数据集的类别不平衡特点，构建基于重采样的C3D网络模型；在上述基础上，提出基于Bagging集成学习的卒中后疲劳识别算法；基于上述算法实现，利用PyQt5设计了一款实时运行、用户友好的卒中后疲劳识别系统，该系统界面如下：
![Image](imgs/用户登陆界面.png)
![Image](imgs/疲劳检测界面.png)
## 运行环境

## 代码运行
### 1 - 运行[generation_process.py](train/generation_process.py)
数据预处理
### 2 - 运行[train.py](train/train.py)
模型训练
### 3 - （可选）运行[inference.py](train/inference.py)
模型测试
### 4 - 运行[main.py](main.py)
卒中后疲劳识别系统入口
