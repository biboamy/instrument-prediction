import numpy as np
import SharedArray as sa
import glob
import os
import time
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../thickstun/')   
from norm_lib import *
import h5py
from config import *
#from ex_librosa_fe import *
#import predict_pitch as pp

def norm(avg, std, data, size):
    avg = np.tile(avg.reshape((1,-1,1,1)),(size[0],1,size[2],size[3]))
    std = np.tile(std.reshape((1,-1,1,1)),(size[0],1,size[2],size[3]))
    data = (data - avg)/std
    return data


def load_te(avg,std):
    path = 'data/ex_data/'
    Xte = np.load(path+'xte.npy')
    Yte = np.load(path+'yte.npy')
    Xte = np.expand_dims(Xte, axis=3)
    tesize = Xte.shape
    Xte = norm(avg, std, Xte, tesize)
    if status == 'har':
        if isE: dim = 1
        else: dim = 3
        Xte_p = load_har_te(h_num) 
        Xte = np.concatenate((Xte, Xte_p),dim)
    
    return Xte, Yte

def load_har_te(h):
    path = 'thickstun/pitch/'
    Xte = np.load(path+'xte_h5.npy')
    Xte = np.expand_dims(Xte, axis=3)

    return Xte
