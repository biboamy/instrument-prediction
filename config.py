#config
'''
Different setting in this experiment
1.CQT only + baseline:
	status = 'inst' 
	model_name = 'JY'  
	isE = False 
2.CQT only + residual:
	status = 'inst' 
	model_name = 'SY'  
	isE = False 
3.HSF-n (Ground truth):
	status = 'har' 
	model_name = 'SY'  
	isE = False 
	data_name = 'gt'
	h_num = (1~5)
4.HSF-n (Estimation):
	status = 'har' 
	model_name = 'SY'  
	isE = False 
	data_name = 'ex'
	h_num = (1~5)
5.HSF-n (Estimation):
	status = 'har' 
	model_name = 'SY'  
	isE = False 
	data_name = 'ex'
	h_num = (1~5)
6.CQT+Pitch(F) (Estimation):
	status = 'har' 
	model_name = 'SY'  
	isE = True 
	data_name = 'ex'
	h_num = 0
7.CQT+Pitch(C) (Estimation):
	status = 'har' 
	model_name = 'SY'  
	isE = False 
	data_name = 'ex'
	h_num = 0
'''
status = 'har' 
data_name = 'ex' 
model_name = 'SY'  
isE = False 
h_num = 5 

saveName = 'har.ex.5.SY' #name of the model to save and load
fre = 88 #number of frequency bin
length = 258 #number of frame in 3 seconds	
num_labels = 7 #number of instrument 
batch_size = 10 

# for evaluation
model_choose = 55 #number of model to load
isPre = True #whether to calculate score matrix (F1-score, Precision and Recall)
isDraw = True #whether to draw the result