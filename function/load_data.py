import numpy as np
import glob
import os
import time
import sys
sys.path.append('../')
from norm_lib import *
from config import *

def norm(avg, std, data, size):
    avg = np.tile(avg.reshape((1,-1,1,1)),(size[0],1,size[2],size[3]))
    std = np.tile(std.reshape((1,-1,1,1)),(size[0],1,size[2],size[3]))
    data = (data - avg)/std
    return data

def load():
    path = 'ex_data/cqt/'
      
    Xte = np.load(path+'xte.npy')
    Xtr = np.load(path+'xtr.npy')
    Yte = np.load(path+'yte.npy')
    Ytr = np.load(path+'ytr.npy')
       
    Xtr = np.expand_dims(Xtr, axis=3)
    Xte = np.expand_dims(Xte, axis=3)

    avg, std = RoW_norm(Xtr, './data/%s%d_avg_std'%('inst',h_num))
    Xtr = norm(avg, std, Xtr, Xtr.shape)
    Xte = norm(avg, std, Xte, Xte.shape)
    
    if status == 'har':
        Xtr_p,Xte_p = load_har(h_num) 
        if isE: 
            if isL:
                Xtr = np.concatenate((Xtr, Xtr_p),3)
                Xte = np.concatenate((Xte, Xte_p),3)
            Xtr = np.concatenate((Xtr, Xtr_p),1)
            Xte = np.concatenate((Xte, Xte_p),1)
        else: 
            Xtr = np.concatenate((Xtr, Xtr_p),3)
            Xte = np.concatenate((Xte, Xte_p),3)
    
     #avg std
    return Xtr, Ytr, Xte, Yte, avg, std

def load_te(avg,std):
    Xte = np.load('ex_data/cqt/xte.npy')
    Yte = np.load('ex_data/cqt/yte.npy')
    Xte = np.expand_dims(Xte, axis=3)
    Xte = norm(avg, std, Xte, Xte.shape)
    if status == 'har':
        if isE: dim = 1
        else: dim = 3
        Xte_p = load_har_te(h_num) 
        Xte = np.concatenate((Xte, Xte_p),dim)
    
    return Xte, Yte

if data_name == 'ex': path = 'pitch_est/'
if data_name == 'gt': path = 'pitch_data_gt/'

def load_har(h):
    Xte = np.load(path+'xte_h'+str(h)+'.npy')
    Xtr = np.load(path+'xtr_h'+str(h)+'.npy')
    Xtr = np.expand_dims(Xtr, axis=3)
    Xte = np.expand_dims(Xte, axis=3)

    return Xtr, Xte

def load_har_te(h):
    Xte = np.load(path+'xte_h'+str(h)+'.npy')
    Xte = np.expand_dims(Xte, axis=3)

    return Xte
