import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
sys.path.append('../fun')
import math
from config import *

class block(nn.Module):
    def __init__(self, inp, out):
        super(block, self).__init__()
        self.bn1 = nn.BatchNorm2d(inp)       
        self.conv1 = nn.Conv2d(inp, out, (3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(out)       
        self.conv2 = nn.Conv2d(out, out, (3,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(out)       

        self.sk = nn.Conv2d(inp, out, (1,1), padding=(0,0))
    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        out += self.sk(x)
        return out

class conv_block(nn.Module):
    def __init__(self, inp, out, kernal, pad, bbin):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(inp, out, kernal, padding=pad)
        self.batch = nn.BatchNorm2d(bbin)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.batch(out)
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dp = nn.Dropout(.0)

        if not isE:
            inp = fre
            if status == 'har':size = 2
            if status == 'inst':size = 1
        else: 
            inp = fre*2
            size = 1
        
        def SY_model():
            fs = (3,1)
            ps = (1,0)
            
            self.head = nn.Sequential(
                nn.BatchNorm2d(inp),       
                nn.Conv2d(inp, fre, (3,size), padding=(1,0)),
                block(fre, fre*2),
                nn.Dropout(p=0.2),
                nn.MaxPool2d((3,1),(3,1)),
                block(fre*2, fre*3),
                nn.Dropout(p=0.2),
                nn.MaxPool2d((3,1),(3,1)),
                block(fre*3, fre*3),
                nn.BatchNorm2d(fre*3),
                nn.ReLU(inplace=True),
                nn.Conv2d( fre*3, num_labels, fs, padding=ps)
            )

        def JY_model():
            self.head = nn.Sequential(
                conv_block(size, 32, (inp, 7), (0,3), 32),
                nn.MaxPool2d((1,3),(1,3)),
                conv_block(32, 32, (1, 7), (0,3), 32),
                nn.MaxPool2d((1,3),(1,3)),
                conv_block(32, 512, (1,1), (0,0), 512),
                conv_block(512, 512, (1,1), (0,0), 512),
                nn.Conv2d(512, num_labels, (1,1), padding=(0,0)),
            )
        if model_name == 'SY': SY_model()
        if model_name == 'JY': JY_model()

    def forward(self, _input, Xavg, Xstd):

        x = _input
        if model_name == 'SY':
            frame_pred = self.head(x)
        if model_name == 'JY':
            frame_pred = self.head(x.permute(0,3,1,2))

        return frame_pred

        

        
        