import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torch.autograd import Variable

# Dataset
class Data2Torch(Dataset):
    def __init__(self, data):

        self.X = data[0]
        self.Y = data[1]

    def __getitem__(self, index):

        mX = torch.from_numpy(self.X[index]).float()
        mY = torch.from_numpy(self.Y[index]).float()

        return mX, mY
    
    def __len__(self):
        return len(self.X)

# lib
def sp_loss(fla_pred, target, gwe):

    we = gwe.cuda()
    wwe = 1
    we *= wwe
    
    loss = 0
    
    for idx, (out, fl_target) in enumerate(zip(fla_pred,target)):
        twe = we.view(-1,1).repeat(1, fl_target.size(1)).type(torch.cuda.FloatTensor)
        ttwe = twe * fl_target.data + (1 - fl_target.data) * wwe
        loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)
        loss += loss_fn(torch.squeeze(out), fl_target)
 
    return loss

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

def num_params(model):
    params = 0
    for i in model.parameters():
        params += i.view(-1).size()[0]
    print ('#params:%d'%(params))
