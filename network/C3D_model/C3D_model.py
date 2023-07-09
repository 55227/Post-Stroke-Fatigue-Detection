import torch
import torch.nn as nn
# from mypath import MyPath

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes=4, pretrained=False,*args,**kwargs):
        try:
            '''Initialize the model'''
            super(C3D, self).__init__()

            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))#输入channel为3，64个3*3*3的卷积核，stride为1
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Linear(8192, 4096)#8192从何得来
            self.fc7 = nn.Linear(4096, 4096)
            self.fc8 = nn.Linear(4096, num_classes)

            self.dropout = nn.Dropout(p=0.5)
            #这里不放入softmax层，原因是我们在本项目中使用的nn.CrossEntropyLoss来计算损失值，而该函数相当于softmax + log + nllloss，为避免多做一次softmax运算，这里不放入softmax层

            self.relu = nn.ReLU()

            self.__init_weight()#初始化权重
            '''知识点：在定义模型之后可以直接初始化权重，Kaiming初始化方法'''
        except Exception as e:
            print(repr(e))
        # if pretrained:#用预训练模型的参数来初始化模型
        #     self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)#调整维度，x的size变为[*,8192]
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

    # def __load_pretrained_weights(self):
    #     """Initialiaze network."""
    #     '''加载预训练模型，利用预训练好的参数初始化模型参数（最后一个全连接层需要从头开始训练）'''
    #     corresp_name = {
    #                     # Conv1
    #                     "features.0.weight": "conv1.weight",
    #                     "features.0.bias": "conv1.bias",
    #                     # Conv2
    #                     "features.3.weight": "conv2.weight",
    #                     "features.3.bias": "conv2.bias",
    #                     # Conv3a
    #                     "features.6.weight": "conv3a.weight",
    #                     "features.6.bias": "conv3a.bias",
    #                     # Conv3b
    #                     "features.8.weight": "conv3b.weight",
    #                     "features.8.bias": "conv3b.bias",
    #                     # Conv4a
    #                     "features.11.weight": "conv4a.weight",
    #                     "features.11.bias": "conv4a.bias",
    #                     # Conv4b
    #                     "features.13.weight": "conv4b.weight",
    #                     "features.13.bias": "conv4b.bias",
    #                     # Conv5a
    #                     "features.16.weight": "conv5a.weight",
    #                     "features.16.bias": "conv5a.bias",
    #                      # Conv5b
    #                     "features.18.weight": "conv5b.weight",
    #                     "features.18.bias": "conv5b.bias",
    #                     # fc6
    #                     "classifier.0.weight": "fc6.weight",
    #                     "classifier.0.bias": "fc6.bias",
    #                     # fc7
    #                     "classifier.3.weight": "fc7.weight",
    #                     "classifier.3.bias": "fc7.bias",
    #                     }
    #     ######加载模型
    #     p_dict = torch.load(MyPath.model_dir())#加载预训练的模型权重文件 p_dict是一个OrderedDict（有序字典），包含corresp_name中的所有键值
    #     # print(p_dict)
    #     s_dict = self.state_dict()#包含网络卷积层和全连接层参数的字典 state_dict（） 是一个Python字典，将每一层映射成它的参数张量。注意只有带有可学习参数的层（卷积层、全连接层等），以及注册的缓存（batchnorm的运行平均值）在state_dict 中才有记录。
    #     #s_dict的键为conv1.weight、conv1.bias、conv2.weight等等
    #     for name in p_dict:
    #         if name not in corresp_name:#判断name是不是corresp_name字典中的key
    #             continue
    #         s_dict[corresp_name[name]] = p_dict[name]
    #     self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():#self.modules为当前网络中模块的迭代器（模块包括conv1、pool1、conv2、pool2、conv3a等等
            if isinstance(m, nn.Conv3d):#模块m是否属于三维卷积模块
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)#Kaiming初始化
                # print(m.bias.shape)#没有对m.bias进行初始化
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters for conv and two fc layers of the net.
        """
        b = [self.conv1, self.conv2, self.conv3a, self.conv3b, self.conv4a, self.conv4b,
             self.conv5a, self.conv5b, self.fc6, self.fc7]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last fc layer of the net.
        """
        b = [self.fc8]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    # print(k)
                    yield k

if __name__ == "__main__":

    net = C3D(pretrained=True)
    net.get_10x_lr_params()

    # outputs = net.forward(inputs)
    # print(outputs.size())