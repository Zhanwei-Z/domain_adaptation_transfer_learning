import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchvision import models
import torch.nn.functional as F

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class ResNet50(nn.Module):
  def __init__(self, new_cls=True, feature_dim=256, class_num=1000):
    super(ResNet50, self).__init__()
    model_resnet = models.resnet50(pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.new_cls = new_cls
    self.feature_dim = feature_dim
    self.features = nn.Linear(model_resnet.fc.in_features, self.feature_dim)
    if new_cls:
        self.fc = nn.Linear(self.feature_dim, class_num)
        self.fc.apply(init_weights)
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x,label=None,weight=None):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    x = self.features(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.feature_dim

  def get_parameters(self):
    if self.new_cls:
        parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size,second=False,max_iter=10000,radius=10.0):
    super(AdversarialNetwork, self).__init__()
    self.radius=radius
    self.ad_layer1 = nn.Linear(in_feature,hidden_size+1)
    self.ad_layer2 = nn.Linear(hidden_size+1,hidden_size+1)
    self.ad_layer3 = nn.Linear(hidden_size+1,1)
    self.sigmoid=nn.Sigmoid()
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter
    self.second=second

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    if self.second==False:
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    else:
        coeff=calc_coeff(self.iter_num, self.high, self.low, self.alpha, 3000)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x=self.ad_layer1(x)
    x=self.ad_layer2(x)
    x=self.ad_layer3(x)
    y=self.sigmoid(x)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


