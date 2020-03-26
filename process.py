# -*- coding: utf-8 -*-
import numpy as np
import librosa
import matplotlib
import matplotlib.pyplot as plt

test_list = ['2303', '2191', '2382', '2628', '2416', '2556', '2298', '1819', '1759', '2106']
inst_lookup = {1:0,41:1,42:2,43:3,72:4,71:5,61:6}
stride = 512

# extract xtr.npy
def extract_data():
	# h: numbers of harmonic
	def logCQT(y,h):
		sr = 44100
		cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=27.5*float(h), n_bins=88, bins_per_octave=12)
		return ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0

	def chunk_data(f):
		s = (44100*3/512)
		num = 88
		xdata = np.transpose(f)
		x = [] 
		length = int(np.ceil((int(len(xdata)/s)+1)*s))

		app = np.zeros((length-xdata.shape[0],xdata.shape[1]))
		xdata = np.concatenate((xdata,app),0)
		for i in range(int(length/s)):
			data=xdata[int(i*s):int(i*s+s)]
			x.append(np.transpose(data[:258,:]))

		return np.array(x)

	# musicnet data path
	train_data = np.load('../instrument-streaming/musicnet/data/musicnet.npz', encoding='bytes', allow_pickle=True)
	x_test,x_train,y_test,y_train = [],[],[],[]
	for key in list(train_data.keys())[:5]:
		x, y = train_data[key]

		# extract CQT and labels for all songs
		x = logCQT(x, 1)
		Yvec = np.zeros((7, x.shape[1]))
		for window in range(Yvec.shape[1]):
			labels = y[window*stride]
			for label in labels:
				if label.data[0] in inst_lookup:
					Yvec[inst_lookup[label.data[0]],window] = 1

		# chunk the data to 10 seconds
		x = chunk_data(x)
		Yvec = chunk_data(Yvec)

		if key in test_list:
			if len(x_test)!=0:
				x_test = np.concatenate((x,x_test),0)
				y_test = np.concatenate((Yvec,y_test),0)
			else:
				x_test = x
				y_test = Yvec
		else:
			if len(x_train)!=0:
				x_train = np.concatenate((x,x_train),0)
				y_train = np.concatenate((Yvec,y_train),0)
			else:
				x_train = x
				y_train = Yvec
	
	np.save('data/ex_data/xte.npy',x_test)
	np.save('data/ex_data/xtr.npy',x_train)
	np.save('data/ex_data/yte.npy',y_test)
	np.save('data/ex_data/ytr.npy',y_train)
	print(x_train.shape, x_test.shape)
extract_data()
