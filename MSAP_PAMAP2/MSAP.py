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
#We use the kaggle platform, so we need to download the Lion optimizer related packages every time.
#!pip install lion-pytorch
#!pip install triton -U --pre
#from lion_pytorch import Lion


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
class Shrinkage_SE(nn.Module):
    def __init__(self, channel,spilt =4,gap_size = 1):
        super(Shrinkage_SE, self).__init__()

        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.attention = nn.Sequential(
            nn.Linear(channel, channel//16),
            nn.ReLU(inplace=True),
            nn.Linear(channel//16, channel), 
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        
        x_raw = x
        x = torch.abs(x)

        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        #attn = self.attention(x)##
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        #attn = attn.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        #sub = x_abs - attn*x

        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x
    
class Shrinkage_ECA(nn.Module):
    def __init__(self, channel, gap_size=1):
        super(Shrinkage_ECA, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.attention = nn.Sequential(
            nn.Linear(channel, channel//16),
            nn.ReLU(inplace=True),
            nn.Linear(channel//16, channel), 
            nn.Sigmoid()
        )
        self.ECA = ECA(channel, k_size=3)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        
        x_raw = x
        x = torch.abs(x)
        #attn = self.attention(x)##
        attn = self.ECA(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        #attn = self.attention(x)##
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        #attn = attn.unsqueeze(2)
        # soft thresholding
        #sub = x_abs - x
        sub = x_abs - attn*x
        
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x
class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# %%


# %%



#model =  g1(mod='ECA',inplanes=3).to(device)
#model =  g2(mod='ECA',inplanes=3).to(device)
#model =  g3(mod='ECA',inplanes=3).to(device)





# %%
def train(model, train_dataloader,val_dataloader, num_epochs=50, lr1 = 0.001,lr2 = 0.003,loss_weight = 0.007):
    # 定义损失函数和优化器.
    criterion1 = nn.CrossEntropyLoss()
#     criterion2 = CenterLoss(12, 12).to(device)
    optimizer1 = Adam(model.parameters(), lr = lr1)
#     optimizer2 = Adam(criterion2.parameters(), lr = lr1)
    #optimizer1 = Lion(model.parameters(), lr = lr1, weight_decay=1.0)
    #optimizer2 = Lion(criterion2.parameters(), lr = lr2, weight_decay=1.0)
    # 定义存储训练和验证结果的列表.
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    # 定义在验证集上表现最好的模型准确率和损失.
    best_val_acc = 0.0
    best_val_loss = float('inf')
    Best_epoch=0
    # 定义存储最佳模型参数的变量.
    best_model_params = None
    # 开始训练模型.
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            model.train()
            # 将输入和标签数据转换为Tensor并放到GPU上（如果有的话）.
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 将梯度清零.
            optimizer1.zero_grad()
#             optimizer2.zero_grad()
            
            # 前向传播.
            outputs = model(inputs)

            # 计算损失和准确率.
            loss = criterion1(outputs, labels)#+loss_weight*criterion2(labels,outputs)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            # 反向传播和更新参数.
            loss.backward()
            optimizer1.step()
#             optimizer2.step()
            # 累计训练损失
            train_loss += loss.item()
            print(">",end="")
        # 计算训练准确率和损失.
        train_acc = 100.0 * train_correct / train_total
        train_loss = train_loss / len(train_dataloader)

        # 在验证集上验证模型.
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            
            for i, (inputs, labels) in enumerate(val_dataloader):
                # 将输入和标签数据转换为Tensor并放到GPU上（如果有的话）.
                model.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播.
                outputs = model(inputs)

                # 计算损失和准确率.
                loss = criterion1(outputs, labels)#+loss_weight*criterion2(labels,outputs,)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # 累计验证损失.
                val_loss += loss.item()

        
        # 计算验证准确率和损失.
        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss / len(val_dataloader)
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            Best_epoch =epoch+1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_params = model.state_dict()
        
        print()
        # 打印训练和验证结果.
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
        
        # 保存训练和验证结果.
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
    print("The best epoch:",Best_epoch,"    Acc:",best_val_acc)
    model.load_state_dict(best_model_params)
    # 返回训练和验证结果.
    return model.eval(),train_loss_list, train_acc_list, val_loss_list, val_acc_list

# %%
def conv3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1(in_planes, out_planes, stride=1):
    """1x1 convolution"""

    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
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
        y = x
        x = self.avg_pool(x)
        x = self.conv1(x.transpose(-1, -2)).transpose(-1, -2)
        out = self.sigmoid(x)
        return out+y

class GatedRes2NetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None,
                 stride=1, scales=4, groups=1, mod='ECA', norm_layer=None):
        super(GatedRes2NetBottleneck, self).__init__()
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
        self.SE = nn.ModuleList([EfficientChannelAttention(bottleneck_planes // scales) for _ in range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales)
                                  for _ in range(scales - 1)])
        self.judge = nn.ModuleList([conv1(bottleneck_planes+2*bottleneck_planes // scales,
                                          bottleneck_planes // scales
                                          ) for _ in range(scales - 2)])
        self.tanh = nn.Tanh()

        self.Shrinkage_SE = Shrinkage_SE(channel = 4*planes,spilt = planes//16)
        self.Shrinkage_ECA = Shrinkage_ECA(channel = 4*planes)
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
        before = []
        att = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            
            elif s == 1:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
                before.append(ys[-1])
                ys[-1] = (self.SE[s - 1](ys[-1]))
            else:
                #gate = self.tanh(self.judge[s - 2](torch.cat([out,xs[s],ys[-1]],dim=1)))
                #ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + gate * ys[-1]))))
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
                before.append(ys[-1])
                att.append(before[-2]+ys[-2])
                ys[-1] = self.SE[s - 1](ys[-1]+att[-1])


        out = torch.cat(ys, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.model =='SE':
            out = self.mod(out)
        elif self.model == 'ECA':
            x = self.mod(out)
            out= out+x
            
#         out = self.Shrinkage_ECA(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out
#########
#
class GatedFCN(nn.Module):
    def __init__(self, layers, maxlength=12, groups=1,
                  width=6,scales=8,mod='ECA', norm_layer=None,inplanes=128):
        super(GatedFCN, self).__init__()
        planes = [int(width * scales * 2 ** i) for i in range(4)]
        #planes=torch.tensor(planes)
        #planes=planes.to('cuda')
        self.pre=inplanes
        self.inplanes=planes[0]
        self.maxlength=maxlength

        self.pre=conv1(inplanes,planes[0])

        self.layer1 = self._make_layer(GatedRes2NetBottleneck, planes[0], layers[0], scales=scales, groups=groups, mod=mod,
                                       norm_layer=norm_layer)
        
        self.layer2 = self._make_layer(GatedRes2NetBottleneck, planes[1], layers[1], stride=2, scales=scales, groups=groups,
                                       mod=mod, norm_layer=norm_layer)

        self.layer3 = self._make_layer(GatedRes2NetBottleneck, planes[2], layers[2], stride=2, scales=scales, groups=groups,
                                       mod=mod, norm_layer=norm_layer)

        self.layer4 = self._make_layer(GatedRes2NetBottleneck, planes[3], layers[3], stride=2, scales=scales, groups=groups,
                                       mod=mod, norm_layer=norm_layer)

        self.roi=nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.15)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.25)
        self.mapping=conv1(in_planes=width * scales * 32 ,out_planes=maxlength)#k=16
        #layernumber=3 k=16, layernumber=4 k=32
        #self.Context_Encoder = Context_Encoder(1024, 512).to(device)
        #self.Context_Encoder = Context_Encoder(12, 6).to(device)
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
        x=  x.to(device)
        x = x.permute(0, 2, 1)

        x=  self.pre(x)
        x1 = self.dropout1(x)
        x1=  self.layer1(x1)
        
        
        x2 = self.dropout2(x1)
        #print(x2.shape)
        x2 = self.layer2(x2)
        
        x3 = self.dropout3(x2)
        x3 = self.layer3(x3)

        x4 = self.dropout4(x3)
        x4 = self.layer4(x4)

        #x4 = self.Context_Encoder.forward_ret(x4)

        x5=  self.roi(x4)
#         x5=  self.roi(x3)
        

        output=self.mapping(x5)

        output=output.squeeze()

        return output

# %%
def g1(mod='None',inplanes=36):
    return GatedFCN([1,2,2,1],mod=mod,inplanes=inplanes)
def g2(mod='None',inplanes=36):
    return GatedFCN([1,2,1,0],mod=mod,inplanes=inplanes)
def g3(mod='None',inplanes=36):
    return GatedFCN([2,3,4,1],mod=mod,inplanes=inplanes)
