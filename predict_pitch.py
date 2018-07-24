import os,sys
import cPickle as pickle
import numpy as np                                       # fast vectors and matrices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # plotting
from scipy.fftpack import fft
from time import time
sys.path.append('./thickstun/lib/') 
sys.path.insert(0,'lib/')
import cf
import diagnostics
import base_model
from sklearn.metrics import average_precision_score
import tensorflow as tf
import os,mmap
import librosa
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#%matplotlib inline

def create_filters(d,k):
    x = np.linspace(0, 2*np.pi, d, endpoint=False)
    wsin = np.empty((1,d,1,k), dtype=np.float32)
    wcos = np.empty((1,d,1,k), dtype=np.float32)
    start_freq = 50.
    end_freq = 6000.
    num_cycles = start_freq*d/44100.
    scaling_ind = np.log(end_freq/start_freq)/k
    window_mask = 1.0-1.0*np.cos(x)
    for ind in range(k):
        wsin[0,:,0,ind] = window_mask*np.sin(np.exp(ind*scaling_ind)*num_cycles*x)
        wcos[0,:,0,ind] = window_mask*np.cos(np.exp(ind*scaling_ind)*num_cycles*x)
            
    return wsin,wcos

class Spectrograms(base_model.Model):
    def __init__(self, *args, **kwargs):
        super(Spectrograms, self).__init__(*args, **kwargs)

    def define_graph(self):
        super(Spectrograms, self).define_graph()
        
        # lvl1 convolutions are shared between regions
        self.k = 512                # lvl1 nodes
        self.d = 4096               # lvl1 receptive field
        
        d2_x = 1          # lvl2 input dims_x
        d2_y = 128          # lvl2 input dims_y
        k2 = 128        # num lvl2 filters
        stride_y = 2    # lvl2 stride
        
        d3_x = 25 # lvl3 input dims_x
        d3_y = 1 # lvl3 input dims_y (fully connected)
        k3 = 4096 # num lvl3 filters

        num_regions  = 1 + (self.window-self.d)/self.stride
        #print 'First layer regions: ({},{})'.format(num_regions,self.k)
        num_regions2_x  = 1 + (num_regions-d2_x)/1
        num_regions2_y = 1 + (self.k-d2_y)/stride_y
        #print 'Second layer regions: ({},{})'.format(num_regions2_x,num_regions2_y)
        num_regions3_x = 1 + (num_regions2_x - d3_x)/1
        num_regions3_y = 1 + (num_regions2_y - d3_y)/1

        wsin,wcos = create_filters(self.d,self.k)

        print '---- Weights ----'
        wscale = .0001
        with tf.variable_scope('parameters'):
            w = tf.Variable(wscale*tf.random_normal([d2_x,d2_y,1,k2],seed=999))
            print 'w',w
            wavg = self.register_weights(w,'w',average=.9998)
            w2 = tf.Variable(wscale*tf.random_normal([d3_x,d3_y,k2,k3],seed=999))
            print 'w2',w2
            w2avg = self.register_weights(w2,'w2',average=.9998)
            beta = tf.Variable(wscale*tf.random_normal([num_regions3_x*num_regions3_y*k3,self.m],seed=999))
            print 'beta',beta
            betaavg = self.register_weights(beta,'beta',average=.9998)

        print '---- Layers ----'
        with tf.variable_scope('queued_model'):
            zx = tf.square(tf.nn.conv2d(self.xq,wsin,strides=[1,1,self.stride,1],padding='VALID')) \
               + tf.square(tf.nn.conv2d(self.xq,wcos,strides=[1,1,self.stride,1],padding='VALID'))
            print 'zx',zx
            z2 = tf.nn.relu(tf.nn.conv2d(tf.log(zx+10e-15),w,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
            print 'z2',z2
            z3 = tf.nn.relu(tf.nn.conv2d(z2,w2,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
            print 'z3',z3
            y = tf.matmul(tf.reshape(z3,[self.batch_size,num_regions3_x*num_regions3_y*k3]),beta)
            print 'y',y
            self.loss = tf.reduce_mean(tf.nn.l2_loss(y-tf.reshape(self.yq,[self.batch_size,self.m])))

        with tf.variable_scope('direct_model'):
            self.zx = tf.square(tf.nn.conv2d(self.xd,wsin,strides=[1,1,self.stride,1],padding='VALID')) \
                    + tf.square(tf.nn.conv2d(self.xd,wcos,strides=[1,1,self.stride,1],padding='VALID'))
            self.z2 = tf.nn.relu(tf.nn.conv2d(tf.log(self.zx+10e-15),wavg,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
            self.z3 = tf.nn.relu(tf.nn.conv2d(self.z2,w2avg,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
            self.y_direct = tf.matmul(tf.reshape(self.z3,[tf.shape(self.xd)[0],num_regions3_x*num_regions3_y*k3]),betaavg)
            self.loss_direct = tf.reduce_mean(tf.nn.l2_loss(self.y_direct-self.yd))

def predict(path):
    labels = None
    try: model.stop()
    except NameError: pass
    model = Spectrograms(labels,checkpoint_path='./thickstun/convnet_experimental2_morelvl3/', outputs=1, window=16384, mmap=True,
                         normalize=True, extended_test_set=False, use_mirex=True, init=False, pitch_transforms=5, jitter=.1,
                         restrict=False,isTest=False)
    print 'finish model loading...'
    for i,f in enumerate(os.listdir('./thickstun/data/records/')[:]):
        if (not os.path.isfile(path+f)):
            try:
                print f + ' complete!'
                mse_test, Yhat, Y, mse_breakdown, avp_breakdown = model.sample_records(int(f[:-4]), 10000, fixed_stride=512)
                np.save(path+f,Yhat.T)
            except Exception as e: print e
        else: print 'exist'


def main(name):
    labels = None
    try: model.stop()
    except NameError: pass
    model = Spectrograms(labels,checkpoint_path='./thickstun/convnet_experimental2_morelvl3/', outputs=1, window=16384, mmap=True,
                         normalize=True, extended_test_set=False, use_mirex=True, init=False, pitch_transforms=5, jitter=.1,
                         restrict=False)
    print 'finish model loading...'

    data, y = librosa.load('./mp3/'+name,44100)
    np.save('./thickstun/tmp/test.npy',data)
    fd = os.open('./thickstun/tmp/test.npy', os.O_RDONLY)
    buff = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
    mse_test, Yhat, Y, mse_breakdown, avp_breakdown = model.sample_records(buff, 10000, fixed_stride=512)
    
    np.save('./thickstun/pitch/'+name[:-4]+'.npy',Yhat)

if __name__ == "__main__":
    main('angel.mp3')
