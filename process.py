# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

start = []
end = []

def get_chunk(i):
	#print(i)
	idx = np.argwhere(i == 1)
	length = np.arange(idx[0],idx[-1])
	space = np.setdiff1d(length,idx,assume_unique=True)
	if len(space) != 0:
		s = idx[0][0]
		e = space[0]
		start.append(s)
		end.append(e)
		i[s:e] = np.zeros((e-s))
		get_chunk(i)
	else:
		start.append(idx[0][0])
		end.append(idx[-1][0])
	return 

for name in os.listdir('result/'):
	data = np.load('result/'+name)
	for idx,i in enumerate(data[:]):
		start = []
		end = []
		j_list = []
		if not np.all(i==0):
			get_chunk(i)
			for s,e in zip(start,end):
				s = round(s/9.3333,2)
				e = round(e/9.3333,2)
				json_data = {'start': str(s), 'end': str(e)}
				j_list.append(json_data)
			print(j_list)
			dir_path = 'json/'+name[:-4]+'/'
			if not os.path.exists(dir_path):
				os.makedirs(dir_path)
			with open(dir_path+str(idx)+'.json', 'w') as outfile:
				json.dump(j_list, outfile)