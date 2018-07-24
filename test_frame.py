import sys
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import f1_score,precision_score,recall_score
date = datetime.datetime.now()

sys.path.append('./function')
from evl import *
from model import * 
from audioset import *
from lib import *
from sklearn.metrics import precision_score

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
def plot(pre,tar,name,_dir,day,isy):

    ax1 = plt.subplot(2, 1, 1)
    ax1.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], minor=True)
    ax1.yaxis.grid(True, which='minor')
    plt.xlim([0, len(pre)])
    plt.yticks(np.arange(7), ('Piano', 'Violin', 'Viola', 'Cello', 'Clarinet', 'Bassoon', 'Horn'))
    if not isy: ax1.get_xaxis().set_visible(False)
    plt.imshow(np.transpose(pre),cmap=plt.cm.binary, interpolation='nearest', aspect='auto')

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], minor=True)
    ax2.yaxis.grid(True, which='minor')
    plt.xlim([0, len(pre)])
    plt.yticks(np.arange(7), ('Piano', 'Violin', 'Viola', 'Cello', 'Clarinet', 'Bassoon', 'Horn'))
    if not isy: ax2.get_xaxis().set_visible(False)
    plt.imshow(np.transpose(tar),cmap=plt.cm.binary, interpolation='nearest', aspect='auto')

    d_path = 'plot/%d%d%d/%s/%s/'%(date.year,date.month,28,_dir,str(isy))
    if not os.path.exists(d_path):
        os.makedirs(d_path)
    plt.savefig(d_path+name+'.png')

def start_test():

    #load model
    model = Net().cuda()
    model.apply(model_init)
    save_dic = torch.load('data/model/ex.hsf.5') 
    model.load_state_dict(save_dic['state_dict'])
    va_th = save_dic['va_th']
    print (np.around(save_dic['evl_metrics'][:,0], decimals=3))
    print (np.around(save_dic['evl_metrics'][:,1], decimals=3))
    print (np.around(save_dic['evl_metrics'][:,2], decimals=3))
    print ('finishing loading model')

    Xavg, Xstd = save_dic['avg'], save_dic['std']
    Xte, Yte = load_te(Xavg.data.cpu().numpy(),Xstd.data.cpu().numpy())
    print ('finishing loading dataset')

    #predict configure
    v_kwargs = {'batch_size': 8, 'num_workers': 10, 'pin_memory': True}
    loader = torch.utils.data.DataLoader(Data2Torch([Xte, Yte]), **v_kwargs)
    all_pred = np.zeros((Yte.shape[0],num_labels,28))
    all_tar = np.zeros((Yte.shape[0],num_labels,28))

    #start predict
    print ('start predicting...')
    model.eval()
    ds = 0
    for idx,_input in enumerate(loader):
        data, target = Variable(_input[0].cuda()), Variable(_input[1].cuda())
        f_pred= model(data, Xavg, Xstd)
        all_tar[ds: ds + len(target)] = target.data.cpu().numpy()
        all_pred[ds: ds + len(target)] = F.sigmoid(torch.squeeze(f_pred)).data.cpu().numpy()
        ds += len(target)

    va_th, evl_matrix, va_out = evl(all_tar, all_pred, va_th)
    print (np.around(evl_matrix[:,0], decimals=3))
    print (np.around(evl_matrix[:,1], decimals=3))
    print (np.around(evl_matrix[:,2], decimals=3))

start_test()
	
