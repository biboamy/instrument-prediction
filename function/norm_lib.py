import numpy as np
import os
import time

def RoW_norm(data, fn, fqs=128):

    if True:
        st = time.time()
        print ('Get std and average')
        
        common_sum = 0
        square_sum = 0

        # remove zero padding
        fle = data.shape[2]
        tfle = 0
        for i in range(len(data)):
            tfle += (data[i].sum(-1).sum(0)!=0).astype('int').sum()
            common_sum += data[i].sum(-1).sum(-1)
            square_sum += (data[i]**2).sum(-1).sum(-1)
            
        common_avg = common_sum / tfle
        square_avg = square_sum / tfle
        
        std = np.sqrt( square_avg - common_avg**2 )

        np.save(fn, [common_avg, std])
        
        print (time.time() - st)
        return common_avg, std


