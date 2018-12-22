import os
from typing import List, Tuple, Dict, Iterator, Union
import numpy as np
from jieba import Tokenizer
from config import POSITIVE, NEGATIVE, \
    VEC_SIZE, MAX_QUERY_WC, MAX_DOC_WC, BIN_NUM, \
    DATA_FILE_PATH, DICTIONARY_PATH, FIGURE_DIR,BATCH_SIZE,MODEL_DIR
from keras.models import load_model
from dataset import DataSequence

model_list=['epoch01_acc0.7526.h5','epoch02_acc0.8323.h5','epoch03_acc0.8398.h5',
			,'epoch04_acc0.7892.h5','epoch05_acc0.8858.h5','epoch06_acc0.7938.h5',
			'epoch07_acc0.7673.h5','epoch08_acc0.8677.h5','epoch09_acc0.8349.h5',
			'epoch10_acc0.7887.h5']

def predict_test(idx:int,model,x):
	strtemp='test_score'+str(idx)+'.txt'
	f_ans = open(os.path.join(DATA_FILE_PATH, strtemp)+,'w')
	predictions: np.ndarray = model.predict_generator(x, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
	print(predictions.shape)
	for i in predictions:
		for j in i:
			f_ans.write(str(j)+'\n')
	f_ans.close()

def predict_valid(idx:int,model,x):
	strtemp='valid_score'+str(idx)+'.txt'
	f_ans = open(os.path.join(DATA_FILE_PATH, strtemp)+,'w')
	predictions: np.ndarray = model.predict_generator(x, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
	for i in predictions:
		for j in i:
			f_ans.write(str(j)+'\n')
	f_ans.close()

def evaluate_valid(idx:int,model,x,f_loss,f_acc):
	evaluations: np.ndarray = model.evaluate_generator(x, steps=1, max_q_size=10, workers=1, pickle_safe=False)
	for i in evaluations:
		f_loss.write(str(i[0])+'\n')		
		f_acc.write(str(i[1])+'\n')

cnt=0
print('make test...')
x_test = DataSequence(dataset = 'test', batch_size = BATCH_SIZE, data_augmentation = False, return_target = False)
print('make valid...')
x_valid = DataSequence(dataset = 'validation', batch_size = BATCH_SIZE, data_augmentation = False, return_target = True)
# x_ = DataSequence(dataset = 'validation', batch_size = BATCH_SIZE, data_augmentation = False, return_target = False)

for file_path in model_list:
	model = load_model(os.path.join(MODEL_DIR, file_path))
	print('for load model '+str(cnt))	
	print('test predict...')
	predict_test(cnt,model,x_test)
	print('valid predict.')
	predict_valid(cnt,model,x_test)
	cnt+=1
