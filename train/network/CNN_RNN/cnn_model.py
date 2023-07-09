import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms


class VisualModel(nn.Module):

    def __init__(self,
                 model_name: str,
                 pretrained: bool = False):
        """ Network model. Gets the model from torchvison and discard the last fully connected layer

        Args:
            model_name: Name of visual model to use.
                model names: https://pytorch.org/docs/stable/torchvision/models.html
        """

        super(VisualModel, self).__init__()

        network = getattr(models, model_name)  # models.model_name
        network = network(pretrained=pretrained)

        self.num_features = self._get_out_feats(model_name)
        network = list(network.children())[:-1]  # discard all fully connected layers
        self.pretrained = pretrained


        self.model = nn.Sequential(*network)  # `*` used to unpack the list

    @classmethod
    def _get_out_feats(cls, name):
        """ Returns the number of features extracted by different visual models
            with image input size (3 x 96 x 96).
        """
        return {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,

            'vgg11': 25088,
            'vgg13': 25088,
            'vgg16': 25088,
            'vgg19': 25088,

            'vgg11_bn': 25088,
            'vgg13_bn': 25088,
            'vgg16_bn': 25088,
            'vgg19_bn': 25088,

            'densenet121': 9216,
            'densenet169': 14976,
            'densenet161': 19872,
            'densenet201': 17280,

            'mobilenet_v2': 11520,

            'resnext50_32x4d': 2048,
            'resnext101_32x8d': 2048,

            'wide_resnet50_2': 2048,
            'wide_resnet101_2': 2048,

            'shufflenet_v2_x0_5': 9216,
            'shufflenet_v2_x1_0': 9216,
            'shufflenet_v2_x1_5': 9216,
            'shufflenet_v2_x2_0': 18432,

            'alexnet': 9216

        }[name]

    def forward(self, x):
        """ Forward pass

        Args:
            x (BS x 3 x H x W)
        """

        return self.model(x)
