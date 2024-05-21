# %%
import torch
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.autograd.function import Function
import numpy as np
import torch.utils.data as Data
import math
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
import time
from torch.optim import Adam
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt

from torch import randperm
from sklearn.utils import resample
import numpy as np
from sklearn.model_selection import train_test_split

import argparse

# %%
# train_x_list = "C:/Users/Savior/Desktop/wisdm/wisdm/x_train.npy"
# train_y_list = "C:/Users/Savior/Desktop/wisdm/wisdm/y_train.npy"
# test_x_list = "C:/Users/Savior/Desktop/wisdm/wisdm/x_test.npy"
# test_y_list = "C:/Users/Savior/Desktop/wisdm/wisdm/y_test.npy"

# %%




# train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
# test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=True)
# val_loader = Data.DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=True)



# %%
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

# %%
if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print("使用GPU训练中：{}".format(torch.cuda.get_device_name()))
else:
    device = torch.device("cpu")
    print("使用CPU训练")

# %%
def conv3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1(in_planes, out_planes, stride=1):
    """1x1 convolution"""

    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.transpose(-1, -2)).transpose(-1, -2)
        out = self.sigmoid(x)
        return out

class GatedRes2NetBottleneck1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None,
                 stride=1, scales=4, groups=1, mod='ECA', norm_layer=None):
        super(GatedRes2NetBottleneck1, self).__init__()
        if planes * groups % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        bottleneck_planes = groups * planes
        self.conv1 = conv1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3(bottleneck_planes // scales,
                                          bottleneck_planes // scales,
                                          groups=groups) for _ in range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales)
                                  for _ in range(scales - 1)])
        self.judge = nn.ModuleList([conv1(bottleneck_planes+2*bottleneck_planes // scales,
                                          bottleneck_planes // scales
                                          ) for _ in range(scales - 2)])
        self.tanh = nn.Tanh()


        self.conv3 = conv1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.model=mod
        if mod == 'ECA':
            self.mod = EfficientChannelAttention(self.expansion*planes)
        elif mod == 'SE':
            self.mod = SEModule(self.expansion*planes)

        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []

        #GATE1
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
            # else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                gate = self.tanh(self.judge[s - 2](torch.cat([out,xs[s],ys[-1]],dim=1)))
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + gate * ys[-1]))))

        # #NO-GATE
        # for s in range(self.scales):
        #     if s == 0:
        #         ys.append(xs[s])
        #     # elif s == 1:
        #     else:
        #         ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
        #     # else:
        #     #     gate = self.tanh(self.judge[s - 2](torch.cat([out,xs[s],ys[-1]],dim=1)))
        #     #     ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + gate * ys[-1]))))


        out = torch.cat(ys, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.model =='SE':
            out = self.mod(out)
        elif self.model == 'ECA':
            x = self.mod(out)
            out= out+x

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out
    
class GatedRes2NetBottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None,
                 stride=1, scales=4, groups=1, mod='ECA', norm_layer=None):
        super(GatedRes2NetBottleneck2, self).__init__()
        if planes * groups % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        bottleneck_planes = groups * planes
        self.conv1 = conv1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3(bottleneck_planes // scales,
                                          bottleneck_planes // scales,
                                          groups=groups) for _ in range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales)
                                  for _ in range(scales - 1)])
        self.judge = nn.ModuleList([conv1(bottleneck_planes+2*bottleneck_planes // scales,
                                          bottleneck_planes // scales
                                          ) for _ in range(scales - 2)])
        self.tanh = nn.Tanh()


        self.conv3 = conv1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.model=mod
        if mod == 'ECA':
            self.mod = EfficientChannelAttention(self.expansion*planes)
        elif mod == 'SE':
            self.mod = SEModule(self.expansion*planes)

        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []

        # #GATE1
        # for s in range(self.scales):
        #     if s == 0:
        #         ys.append(xs[s])
        #     elif s == 1:
        #     # else:
        #         ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
        #     else:
        #         gate = self.tanh(self.judge[s - 2](torch.cat([out,xs[s],ys[-1]],dim=1)))
        #         ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + gate * ys[-1]))))

        #NO-GATE
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            # elif s == 1:
            else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            # else:
            #     gate = self.tanh(self.judge[s - 2](torch.cat([out,xs[s],ys[-1]],dim=1)))
            #     ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + gate * ys[-1]))))


        out = torch.cat(ys, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.model =='SE':
            out = self.mod(out)
        elif self.model == 'ECA':
            x = self.mod(out)
            out= out+x

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out

class GatedFCN2(nn.Module):
    def __init__(self, layers, maxlength=18, groups=1,
                  width=8,scales=4,mod='ECA', norm_layer=None,inplanes=128):
        super(GatedFCN2, self).__init__()
        planes = [int(width * scales * 2 ** i) for i in range(4)]
        #planes=torch.tensor(planes)
        #planes=planes.to('cuda')
        self.pre=inplanes
        self.inplanes=planes[0]
        self.maxlength=maxlength

        self.pre=conv1(inplanes,planes[0])

        self.layer1 = self._make_layer(GatedRes2NetBottleneck2, planes[0], layers[0], scales=scales, groups=groups, mod=mod,
                                       norm_layer=norm_layer)
        self.layer2 = self._make_layer(GatedRes2NetBottleneck2, planes[1], layers[1], stride=2, scales=scales, groups=groups,
                                       mod=mod, norm_layer=norm_layer)
        self.layer3 = self._make_layer(GatedRes2NetBottleneck2, planes[2], layers[2], stride=2, scales=scales, groups=groups,
                                       mod=mod, norm_layer=norm_layer)
        self.layer4 = self._make_layer(GatedRes2NetBottleneck2, planes[3], layers[3], stride=2, scales=scales, groups=groups,
                                       mod=mod, norm_layer=norm_layer)
        self.roi=nn.AdaptiveAvgPool1d(output_size=1)
        self.mapping=conv1(in_planes=1024,out_planes=maxlength)

        self.preluip1 = nn.PReLU()
    def _make_layer(self, block, planes, block_num, stride=1, scales=4, groups=1, mod='ECA', norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, downsample, stride=stride, scales=scales, groups=groups, mod=mod,
                            norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, planes, scales=scales, groups=groups, mod=mod, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self,x):
        x1=  x.to(device)
        x1 = x1.permute(0, 2, 1)
        x1=  self.pre(x1).to(device)
        x1=  self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5=  self.roi(x4)
        output=self.mapping(x5)
        output=output.squeeze()

        return output
    
class GatedFCN3(nn.Module):
    def __init__(self, layers, maxlength=18, groups=1,
                  width=8,scales=4,mod='ECA', norm_layer=None,inplanes=128):
        super(GatedFCN3, self).__init__()
        planes = [int(width * scales * 2 ** i) for i in range(4)]
        #planes=torch.tensor(planes)
        #planes=planes.to('cuda')
        self.pre=inplanes
        self.inplanes=planes[0]
        self.maxlength=maxlength

        self.pre=conv1(inplanes,planes[0])

        self.layer1 = self._make_layer(GatedRes2NetBottleneck1, planes[0], layers[0], scales=scales, groups=groups, mod=mod,
                                       norm_layer=norm_layer)
        self.layer2 = self._make_layer(GatedRes2NetBottleneck1, planes[1], layers[1], stride=2, scales=scales, groups=groups,
                                       mod=mod, norm_layer=norm_layer)
        self.layer3 = self._make_layer(GatedRes2NetBottleneck1, planes[2], layers[2], stride=2, scales=scales, groups=groups,
                                       mod=mod, norm_layer=norm_layer)
        self.layer4 = self._make_layer(GatedRes2NetBottleneck1, planes[3], layers[3], stride=2, scales=scales, groups=groups,
                                       mod=mod, norm_layer=norm_layer)
        self.roi=nn.AdaptiveAvgPool1d(output_size=1)
        self.mapping=conv1(in_planes=1024,out_planes=maxlength)

        self.preluip1 = nn.PReLU()
    def _make_layer(self, block, planes, block_num, stride=1, scales=4, groups=1, mod='ECA', norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, downsample, stride=stride, scales=scales, groups=groups, mod=mod,
                            norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, planes, scales=scales, groups=groups, mod=mod, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self,x):
        x1=  x.to(device)
        x1 = x1.permute(0, 2, 1)
        x1=  self.pre(x1).to(device)
        x1=  self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5=  self.roi(x4)
        output=self.mapping(x5)
        output=output.squeeze()

        return output

# %%
def g1(mod='ECA',inplanes=3):
    # return GatedFCN([2,3,2,1],mod=mod,inplanes=inplanes)
    return GatedFCN2([3,2,1,1],mod=mod,inplanes=inplanes)#SE
def g2(mod='ECA',inplanes=3):
    # return GatedFCN([2,3,2,1],mod=mod,inplanes=inplanes)
    return GatedFCN3([3,2,1,1],mod=mod,inplanes=inplanes)#GATE
def g3(mod='ECA',inplanes=3):
    return GatedFCN2([3,4,6,3],mod=mod,inplanes=inplanes)



# %%
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1   = nn.Conv1d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.ca1 = ChannelAttention(128)
        self.sa1 = SpatialAttention()

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        self.ca2 = ChannelAttention(256)
        self.sa2 = SpatialAttention()

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(True),
            nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(True)
        )
        self.ca3 = ChannelAttention(384)
        self.sa3 = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(1152, 18)
        )



    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.layer1(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        # print(x.shape)
        x = self.layer2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        # print(x.shape)
        x = self.layer3(x)
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        # print(x.shape)
        return x

# %%
model =  g1(mod='SE',inplanes=36).to(device)
#model =  g2(mod='SE',inplanes=36).to(device)#GATE
#model =  g3(mod='ECA',inplanes=3).to(device)
# model =  cnn().to(device)

# %%



