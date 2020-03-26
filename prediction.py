#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import librosa
import matplotlib
from torch.autograd import Variable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import torch.nn.init as init
from torch.utils.data import Dataset
date = datetime.datetime.now()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_choose = sys.argv[2]

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

        if model_choose in ['residual']:size = 1
        if model_choose in ['ex.hsf.1','ex.hsf.2','ex.hsf.3','ex.hsf.4','ex.hsf.5']:size=2
        
        fre = 88
        num_labels = 7
        
        def SY_model():
            fs = (3,1)
            ps = (1,0)
            
            self.head = nn.Sequential(
                nn.BatchNorm2d(fre),       
                nn.Conv2d(fre, fre, (3,size), padding=(1,0)),
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
                conv_block(1, 32, (fre, 7), (0,3), 32),
                nn.MaxPool2d((1,3),(1,3)),
                conv_block(32, 32, (1, 7), (0,3), 32),
                nn.MaxPool2d((1,3),(1,3)),
                conv_block(32, 512, (1,1), (0,0), 512),
                conv_block(512, 512, (1,1), (0,0), 512),
                nn.Conv2d(512, num_labels, (1,1), padding=(0,0)),
            )

        if model_choose in ['baseline']:JY_model()
        if model_choose in ['residual','ex.hsf.1','ex.hsf.2','ex.hsf.3','ex.hsf.4','ex.hsf.5']:SY_model()

    def forward(self, _input, Xavg, Xstd):
        x = _input
        if model_choose in ['baseline']: 
            x=x.permute(0,3,2,1).contiguous()
            frame_pred = self.head(x)
        if model_choose in ['residual','ex.hsf.1','ex.hsf.2','ex.hsf.3','ex.hsf.4','ex.hsf.5']: 
            x=x.permute(0,2,1,3).contiguous()
            frame_pred = self.head(x)#.permute(0,2,3,1)  

        return frame_pred


def norm(avg, std, data, size):
    avg = np.tile(avg.reshape((1,1,-1,1)),(size[0],size[1],1,size[3]))
    std = np.tile(std.reshape((1,1,-1,1)),(size[0],size[1],1,size[3]))
    data = (data - avg)/std
    return data
    
def load_te_mp3(name,avg, std):
    def logCQT(file,h):
        sr = 44100
        y, sr = librosa.load(file,sr=sr)
        cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=27.5*float(h), n_bins=88, bins_per_octave=12)
        return ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0

    def chunk_data(f):
        s = (44100*10/512)
        num = 88
        xdata = np.transpose(f)
        x = [] 
        length = int(np.ceil((int(len(xdata)/s)+1)*s))
        app = np.zeros((length-xdata.shape[0],xdata.shape[1]))
        xdata = np.concatenate((xdata,app),0)
        for i in range(int(length/s)):
            data=xdata[int(i*s):int(i*s+s)]
            x.append(np.transpose(data[:816,:]))

        return np.array(x)

    def pitch_shift(inp, shift):
        data = np.roll(inp, shift, axis=0)
        data[:shift,:] = np.zeros((shift,inp.shape[1]))
        return data

    def shift_row(inp):
        data = np.roll(inp, 16, axis=1)
        data[:,:16] = np.zeros((inp.shape[0],16))
        return data

    def norm_in_range(array, mi, ma):
        ran = array.max() - array.min()
        array = (array - array.min()) / ran
        ran2 = ma - mi
        normalized = (array*ran2) + mi
        return normalized

    x = logCQT('mp3/'+name,1)
    x = chunk_data(x)
    x = np.transpose(np.expand_dims(x, axis=3), (0, 2, 1, 3))
    x = norm(avg, std, x, x.shape)
    if model_choose in ['ex.hsf.1','ex.hsf.2','ex.hsf.3','ex.hsf.4','ex.hsf.5']:
        pitch = np.load('./thickstun/pitch/'+name[:-4]+'.npy').T
        har_map = []
        har_map.append(chunk_data(shift_row(pitch[21:109,:])))
        for h in [12,19,24,28,31][:int(model_choose[-1])]:
            har_map.append(chunk_data(shift_row(pitch_shift(pitch[21:109,:],h))))
        ydata = np.array(har_map).sum(0)
        ydata = norm_in_range(ydata,0.0, 1)
        ydata = np.expand_dims(ydata, axis=3)
        x = np.concatenate((x[:ydata.shape[0],:,:,:], np.transpose(ydata,(0,2,1,3))),3)
    return x

def model_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif classname.find('Linear') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
       # init.constant(m.bias, 0)

# Dataset
class Data2Torch(Dataset):
    def __init__(self, data):
        self.X = data[0]

    def __getitem__(self, index):
        mX = torch.from_numpy(self.X[index]).float()
        return mX
    
    def __len__(self):
        return len(self.X)

def main(argv):
    #name = argv[0]
    name = argv[1]

    save_dic = torch.load('./data/model/'+model_choose,encoding='latin1') 
    va_th = save_dic['va_th']

    #load test dataset
    Xavg, Xstd = save_dic['avg'], save_dic['std']
    Xte = load_te_mp3(name,Xavg.data.cpu().numpy(), Xstd.data.cpu().numpy())
    print ('finishing loading dataset')

    #load model
    model = Net().cuda()
    model.apply(model_init)
    model_dict = model.state_dict()
    pretrained_dict1 = {k: v for k, v in save_dic['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict1) 
    model.load_state_dict(model_dict)
    print ('finishing loading model')

    #predict configure
    v_kwargs = {'batch_size': 8, 'num_workers': 10, 'pin_memory': True}
    loader = torch.utils.data.DataLoader(Data2Torch([Xte]), **v_kwargs)

    all_pred = np.zeros((Xte.shape[0],7,int(Xte.shape[1]/9)))
    #start predict
    print ('start predicting...')
    model.eval()
    ds = 0
    for idx,_input in enumerate(loader):
        data = Variable(_input.cuda())
        pred = model(data, Xavg, Xstd)
        all_pred[ds: ds + len(data)] = F.sigmoid(pred.squeeze()).data.cpu().numpy()
        ds += len(data)

    for i, (p) in enumerate(all_pred):
        p = p - np.expand_dims(va_th,1)
        p[p>0] = 1
        p[p<0] = 0
        if i == 0: pre = p
        else: pre = np.append(pre, p, axis=1)
    pre = np.vstack(pre)
    np.save('result/'+name[:-3]+'npy',pre)
    
    plt.figure(figsize=(10,3))
    plt.yticks(np.arange(8), ('Piano', 'Violin', 'Viola', 'Cello', 'Clarinet', 'Bassoon', 'Horn'))
    plt.imshow(pre,cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.savefig('plot/'+name+'.png')
    print ('finish!')
    
if __name__ == "__main__":
    main(sys.argv)
    