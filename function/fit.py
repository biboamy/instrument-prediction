import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import sys
import numpy as np
from evl import *
from lib import *
from config import *

class Trainer:
    def __init__(self, model, lr, epoch, save_fn, avg, std):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        
        Xavg = torch.from_numpy(avg) 
        Xstd = torch.from_numpy(std) 
        self.Xavg, self.Xstd = Variable(Xavg.cuda()), Variable(Xstd.cuda())
        
    def Tester(self, loader, b_size,th):

        all_pred = np.zeros((b_size, num_labels, int(length/9)))
        all_tar = np.zeros((b_size, num_labels, int(length/9)))
        
        self.model.eval()
        ds = 0
        for idx,_input in enumerate(loader):
            data, target = Variable(_input[0].cuda()),Variable(_input[1].cuda())            
            frame_pred = self.model(data, self.Xavg, self.Xstd)
            target = F.max_pool1d(target,9,9)

            all_tar[ds: ds + len(target)] = target.data.cpu().numpy()
            all_pred[ds: ds + len(target)] = F.sigmoid(torch.squeeze(frame_pred)).data.cpu().numpy()
            ds += len(target)
        va_th, evl_matrix, va_out = evl(all_tar, all_pred,th)
        return va_th, evl_matrix, va_out
    
    def fit(self, tr_loader, va_loader, we):
        st = time.time()

        save_dict = {}
        save_dict['tr_loss'] = []

        for e in range(1, self.epoch+1):

            lr = self.lr ** ((e/(50*1))+1) 
            loss_total = 0
            print ('\n==> Training Epoch #%d lr=%4f'%(e, lr))
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        
            # Training
            for batch_idx, _input in enumerate(tr_loader):
                data, target = Variable(_input[0].cuda()), Variable(_input[1].cuda())
            
                #start feed in                
                frame_pred  = self.model(data, self.Xavg, self.Xstd)

                #counting loss
                loss = sp_loss(frame_pred, target, we)
                loss_total += loss.data
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                #frush the board
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d]\tLoss %4f\tTime %d'
                        %(e, self.epoch, batch_idx+1, len(tr_loader),
                            loss.data, time.time() - st))
                sys.stdout.flush()
 
            print ('\n')
            print (loss_total/len(tr_loader))
            print (self.save_fn)
            va_th, evl_matrix, va_out = self.Tester(tr_loader, len(tr_loader.dataset),[])
            va_th, evl_matrix, va_out = self.Tester(va_loader, len(va_loader.dataset),va_th)
            print (np.around(evl_matrix[:,0], decimals=3))
            print (np.around(evl_matrix[:,1], decimals=3))
            print (np.around(evl_matrix[:,2], decimals=3))

            save_dict['state_dict'] = self.model.state_dict()
            save_dict['va_out'] = va_out
            save_dict['va_th'] = va_th
            save_dict['evl_metrics'] = evl_matrix
            save_dict['avg'] = self.Xavg
            save_dict['std'] = self.Xstd
            torch.save(save_dict, self.save_fn+'_e_%d'%(e))