#zmq
from network.C3D_model.C3D_model import C3D
from network.CNN_RNN.cnn_rnn_model import VisualRNNModel

def get_model(modelName:str="C3D",*args,**kwargs):
    return {
        'C3D':C3D,
        'VisualRNNModel':VisualRNNModel
    }[modelName](*args,**kwargs)
