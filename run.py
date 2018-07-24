import torch.optim as optim
import datetime
date = datetime.datetime.now()
import sys
sys.path.append('./function')
from lib import *
from fit import *
from model import *  
from audioset import *
from config import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' # change

def get_weight(Ytr):
	mp = Ytr[:].sum(0).sum(0)
	mmp = mp.astype(np.float32) / mp.sum()
	cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
	inverse_feq = torch.from_numpy(cc)
	return inverse_feq

out_model_fn = './data/model/%s/'%(saveName)
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

# load data
Xtr,Ytr,Xte,Yte,avg,std = load()
print 'finishing data loading...'

# Build Dataloader
t_kwargs = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True,'drop_last': True}
v_kwargs = {'batch_size': batch_size, 'num_workers': 10, 'pin_memory': True}
tr_loader = torch.utils.data.DataLoader(Data2Torch([Xtr[:], Ytr[:]]), shuffle=True, **t_kwargs)
va_loader = torch.utils.data.DataLoader(Data2Torch([Xte, Yte]), **v_kwargs)
print 'finishing data building...'

#Construct Model
model = Net().cuda()
model.apply(model_init)
print model
num_params(model)
print 'batch_size:%d num_labels:%d'%(batch_size, num_labels)
print 'Dataset:' + data_name
print 'Xtr:' + str(Xtr.shape)
print 'Xte:' + str(Xte.shape)
print 'Ytr:' + str(Ytr.shape)
print 'Yte:' + str(Yte.shape)
inverse_feq = get_weight(Ytr.transpose(0,2,1))

#Start training
Trer = Trainer(model, 0.01, 100, out_model_fn, avg,std)
Trer.fit(tr_loader, va_loader,inverse_feq)

