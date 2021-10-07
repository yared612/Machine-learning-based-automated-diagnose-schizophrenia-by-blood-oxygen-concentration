import numpy as np
import math
import pandas as pd
import glob
import matplotlib.pyplot as plt
from random import choice
#from model import *
from LSTM import *
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

test_file   = 'D:\\HCSZ\\all(WCST)\\ch1_16\\test'
class_ = glob.glob(test_file + '\\*')
class_name = []
for name in class_:
    class_name.append(name.split('\\')[-1])
    
def read_data(data_file):
    filename = glob.glob(data_file + '\\*.csv')
    im_name = []
    for i in filename:
        n = i.split('\\')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    aa = []
    for file in im_name:
        data = pd.read_csv(data_file + '\\' + file + '.csv')
        data = np.array(data)[:,1]
        aa.append(data)            
    for k in range(0,len(aa)):
        ss = np.reshape(aa[k],(1,)+aa[k].shape).T
        data_split_out.append(np.reshape(ss,(1,)+ss.shape))
    return data_split_out

def nor(list_,mode = 'nor'):
    out = []
    if mode == 'nor':
        for l in list_:
            l = l/.3
#            l = (l - l.min())/(l.max() - l.min())
            out.append(l)
    if mode == 'std':
        for l in list_:
            l = (l - l.mean())/l.std()
            out.append(l)
    return out
def pre_argmax(predict):
    ans = []
    for i in range(0,len(predict)):
        yy = np.argmax(predict[i])
        ans.append(yy)
    return ans

def pred(X):
    predict = []
    for i in range(0,len(X)):
        y_predict = model.predict(X[i])
        predict.append(y_predict)
    return predict

gx = read_data(test_file + '\\' + class_name[0] )
bx = read_data(test_file + '\\' + class_name[1] )
good_y = np.zeros([len(gx)])
bad_y  = np.ones([len(bx)])
train_y_z = np.concatenate([good_y,bad_y])
gx.extend(bx)
X, y = nor(gx,'nor') , train_y_z

model = class_model()
model.load_weights('D:\\HCSZ\\saved_models\\lstmCNN(all(WCST)).h5')
predict = []
predict = pred(X)
ans = pre_argmax(predict)
    

