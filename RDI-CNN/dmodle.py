#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.nn import functional as F

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)

class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)

class ResNet1(nn.Module):
    def __init__(self):
        super(ResNet1, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )   
    def forward(self, x):
        x = self.prepare(x)
        x = self.layer1(x)
        return x

class ResNet2(nn.Module):
    def __init__(self):
        super(ResNet2, self).__init__()
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1)
        )
    def forward(self, x):
        x = self.layer2(x)
        return x

class ResNet3(nn.Module):
    def __init__(self):
        super(ResNet3, self).__init__()
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1)
        )
    def forward(self, x):
        x = self.layer3(x)
        return x

class ResNet4(nn.Module):
    def __init__(self):
        super(ResNet4, self).__init__()
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, x):
        x = self.layer4(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return x

class GAPFC(nn.Module):
    def __init__(self, classes_num):
        super(GAPFC, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, classes_num)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class ConvFusion(nn.Module):
    def __init__(self, in_channels):
        super(ConvFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.gap(x)
        return x


# In[2]:


class MWBlock1(nn.Module):
    def __init__(self):
        super(MWBlock1, self).__init__()
        self.conv1 = ConvFusion(64)
        self.conv2 = ConvFusion(64)
        self.conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.layer = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1)
        )
        self.soft = nn.Softmax(dim=1)
        self.linear_layer = nn.Linear(2, 128)
        
    def forward(self, x1, x2):
        ini_x = torch.cat((x1, x2), dim = 1)
        ini_x = self.conv(ini_x)
        ini_x = self.layer(ini_x)
        
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = torch.cat((x1, x2), dim = 1)
        x = self.soft(x)
        x = x.squeeze()
        x = self.linear_layer(x)
        x = F.relu(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = ini_x*x
        return x

class MWBlock2(nn.Module):
    def __init__(self):
        super(MWBlock2, self).__init__()
        self.conv1 = ConvFusion(128)
        self.conv2 = ConvFusion(128)
        self.conv3 = ConvFusion(128)
        self.conv4 = ConvFusion(128)
        self.conv = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
        self.layer = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1)
        )
        self.soft = nn.Softmax(dim=1)
        self.linear_layer = nn.Linear(3, 256)
        
    def forward(self, x1, x2, x3):
        ini_x = torch.cat((x1, x2, x3), dim = 1)
        ini_x = self.conv(ini_x)
        ini_x = self.layer(ini_x)
        
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x = torch.cat((x1, x2, x3), dim = 1)
        x = self.soft(x)
        x = x.squeeze()
        x = self.linear_layer(x)
        x = F.relu(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = ini_x*x
        return x

class MWBlock3(nn.Module):
    def __init__(self):
        super(MWBlock3, self).__init__()
        self.conv1 = ConvFusion(256)
        self.conv2 = ConvFusion(256)
        self.conv3 = ConvFusion(256)
        self.conv4 = ConvFusion(256)
        self.conv = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)
        self.linear_layer = nn.Linear(3, 256)
        
    def forward(self, x1, x2, x3):
        ini_x = torch.cat((x1, x2, x3), dim = 1)
        ini_x = self.conv(ini_x)
        
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x = torch.cat((x1, x2, x3), dim = 1)
        x = self.soft(x)
        x = x.squeeze()
        x = self.linear_layer(x)
        x = F.relu(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = ini_x*x
        return x

