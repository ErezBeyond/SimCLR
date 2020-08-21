import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, pretrained=False):
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                            "resnet50": models.resnet50(pretrained=pretrained)}
        super(ResNetSimCLR, self).__init__()

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features
        self.conv_net = nn.Sequential(*list(resnet.children())[:-2])
        self.pooling = list(resnet.children())[-2]
        self.features = nn.Sequential(self.conv_net, self.pooling)

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def project(self, h):
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return x

    def forward(self, x):
        c = self.conv_net(x)
        h = self.pooling(c)
        h = h.squeeze()
        z = self.project(h)
        return h, z, c
def get_resnet_goemetric_transform():
    H = np.array([[1/32, 0, 0],
                  [0, 1/32, 0],
                  [0, 0, 1]])
    return H