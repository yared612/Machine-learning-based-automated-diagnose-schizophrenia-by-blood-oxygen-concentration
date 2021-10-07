import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import roc_auc_score
#from xgboost import plot_importance
from matplotlib import pyplot
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
import numpy as np
import math
import pandas as pd
import glob
import matplotlib.pyplot as plt
from random import choice


train_file = 'D:\\HCSZ\\all(TMT)v\\ch1_16\\train'
val_file   = 'D:\\HCSZ\\all(TMT)v\\ch1_16\\val'
path = 'D:\\HCSZ\\'
class_ = glob.glob(train_file + '\\*')
class_name = []
for name in class_:
    class_name.append(name.split('\\')[-1])
#WCST & VFT
def data_aug1(data_file,mm,cc,crop_number = 10):
    filename = glob.glob(data_file + '*.csv')
    im_name = []
    for i in filename:
        n = i.split('\\')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    for file in im_name:
        data = pd.read_csv(data_file + file + '.csv')
#        data1 = pd.read_csv(data_file + 'B\\' + file + '.csv')
        data = np.array(data)[:,cc]
#        data1 = np.array(data1)[:,5]
#        data = np.concatenate((data, data1), axis=0)
        data = np.reshape(data,data.shape+(1,))
#        pg = pd.read_csv(data_file + 'pn'+ '/' + file + '.csv')
#        data.loc[data.shape[0]+1] = pg.iat[0,1]
#        data = np.array(data)
#        number = int(len(data)/crop_number)
        data_crop  = []    
        for chan in range(0,data.shape[1]):
            for num in range(0,len(data),crop_number):
                data_crop1 = data[num:num+crop_number,chan]
                random     = choice(data_crop1)
                data_crop.append(random)
        c =[]
        k = math.ceil(len(data_crop)/data.shape[1])            
        for i in range(0,len(data_crop),k):
            c0 = data_crop[i:i+k]
            x0 = np.linspace(0,len(c0)-1,len(c0))
            x1 = np.linspace(0,len(c0)-1,mm)
            y0 = np.interp(x1,x0,c0)
            c.append(y0)            
        c = np.stack(c,axis=-1)
        data_split_out.append(c.T)
    return data_split_out
def frequency(data_file,num,crop_number,mm,cc):
    final = []
    for i in range(1,num):
        x = data_aug1(data_file,mm,cc,crop_number)
        final+=x     
    return final

def nor(list_,mode = 'nor'):
    out = []
    if mode == 'nor':
        for l in list_:
            l = l/.5
#            l = (l - l.min())/(l.max() - l.min())
            out.append(l)
    if mode == 'std':
        for l in list_:
            l = (l - l.mean())/l.std()
            out.append(l)
    return out
def read_data(data_file,cc):
    filename = glob.glob(data_file +'\\*.csv')
    im_name = []
    for i in filename:
        n = i.split('\\')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    aa = []
    for file in im_name:
        data = pd.read_csv(data_file + file + '.csv')
#        data1 = pd.read_csv(data_file + 'B\\' + file + '.csv')
        data = np.array(data)[:,cc]
#        data1 = np.array(data1)[:,13]
#        data = np.concatenate((data, data1), axis=0)
        data = np.reshape(data,data.shape+(1,))
        aa.append(data.T)            
    for k in range(0,len(aa)):
        data_split_out.append(aa[k])
    return data_split_out
#---------------------------------------------------------------------------------------------------
#WCST mack
#    1
wgood_x11 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,5)
wgood_x21 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,5)
wgood_x31 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,5)
wgood_X1  = wgood_x11 + wgood_x21 + wgood_x31
wbad_x11 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,5)
wbad_x21 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,5)
wbad_x31 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,5)
wbad_X1 = wbad_x11 + wbad_x21 + wbad_x31
wval_good_x11 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,5)
wval_good_x21 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,5)
wval_good_x31 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,5)
wval_good_X1 = wval_good_x11 + wval_good_x21 + wval_good_x31
wval_bad_x11 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,5)
wval_bad_x21 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,5)
wval_bad_x31 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,5)
wval_bad_X1 = wval_bad_x11 + wval_bad_x21 + wval_bad_x31
wgood_y1 = np.zeros(len(wgood_X1))
wbad_y1  = np.ones(len(wbad_X1))
wtrain_y_z1 = np.concatenate([wgood_y1,wbad_y1])
wval_good_y1 = np.zeros([len(wval_good_X1)])
wval_bad_y1  = np.ones([len(wval_bad_X1)])
wval_y_z1 = np.concatenate([wval_good_y1,wval_bad_y1])
wgood_X1.extend(wbad_X1), wval_good_X1.extend(wval_bad_X1)
wX1, wy1, wval_X1, wval_y1 = nor(wgood_X1,'nor') , wtrain_y_z1 , nor(wval_good_X1,'nor') , wval_y_z1

wss1 = wX1[0]
wdf1 = pd.DataFrame(wss1)
for i in range (1,len(wX1)):
    wsss1 = wX1[i]
    wdf1 = pd.concat([wdf1, pd.DataFrame(wsss1)], axis=0)
#df = df.dropna(axis=0,how='any')
wvv1 = wval_X1[0]
wdf11 = pd.DataFrame(wvv1)
for j in range (1,len(wval_X1)):
    wvvv1 = wval_X1[j]
    wdf11 = pd.concat([wdf11, pd.DataFrame(wvvv1)], axis=0)
#   2
wgood_x12 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,5)
wgood_x22 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,5)
wgood_x32 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,5)
wgood_X2  = wgood_x12 + wgood_x22 + wgood_x32
wbad_x12 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,5)
wbad_x22 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,5)
wbad_x32 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,5)
wbad_X2 = wbad_x12 + wbad_x22 + wbad_x32
wval_good_x12 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,5)
wval_good_x22 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,5)
wval_good_x32 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,5)
wval_good_X2 = wval_good_x12 + wval_good_x22 + wval_good_x32
wval_bad_x12 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,5)
wval_bad_x22 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,5)
wval_bad_x32 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,5)
wval_bad_X2 = wval_bad_x12 + wval_bad_x22 + wval_bad_x32
wgood_y2 = np.zeros(len(wgood_X2))
wbad_y2  = np.ones(len(wbad_X2))
wtrain_y_z2 = np.concatenate([wgood_y2,wbad_y2])
wval_good_y2 = np.zeros([len(wval_good_X2)])
wval_bad_y2  = np.ones([len(wval_bad_X2)])
wval_y_z2 = np.concatenate([wval_good_y2,wval_bad_y2])
wgood_X2.extend(wbad_X2), wval_good_X2.extend(wval_bad_X2)
wX2, wy2, wval_X2, wval_y2 = nor(wgood_X2,'nor') , wtrain_y_z2 , nor(wval_good_X2,'nor') , wval_y_z2

wss2 = wX2[0]
wdf2 = pd.DataFrame(wss2)
for i in range (1,len(wX2)):
    wsss2 = wX2[i]
    wdf2 = pd.concat([wdf2, pd.DataFrame(wsss2)], axis=0)
#df = df.dropna(axis=0,how='any')
wvv2 = wval_X2[0]
wdf12 = pd.DataFrame(wvv2)
for j in range (1,len(wval_X2)):
    wvvv = wval_X2[j]
    wdf12 = pd.concat([wdf12, pd.DataFrame(wvvv)], axis=0)
#    3
wgood_x13 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,5)
wgood_x23 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,5)
wgood_x33 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,5)
wgood_X3  = wgood_x13 + wgood_x23 + wgood_x33
wbad_x13 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,5)
wbad_x23 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,5)
wbad_x33 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,5)
wbad_X3 = wbad_x13 + wbad_x23 + wbad_x33
wval_good_x13 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,5)
wval_good_x23 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,5)
wval_good_x33 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,5)
wval_good_X3 = wval_good_x13 + wval_good_x23 + wval_good_x33
wval_bad_x13 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,5)
wval_bad_x23 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,5)
wval_bad_x33 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,5)
wval_bad_X3 = wval_bad_x13 + wval_bad_x23 + wval_bad_x33
wgood_y3 = np.zeros(len(wgood_X3))
wbad_y3  = np.ones(len(wbad_X3))
wtrain_y_z3 = np.concatenate([wgood_y3,wbad_y3])
wval_good_y3 = np.zeros([len(wval_good_X3)])
wval_bad_y3  = np.ones([len(wval_bad_X3)])
wval_y_z3 = np.concatenate([wval_good_y3,wval_bad_y3])
wgood_X3.extend(wbad_X3), wval_good_X3.extend(wval_bad_X3)
wX3, wy3, wval_X3, wval_y3 = nor(wgood_X3,'nor') , wtrain_y_z3 , nor(wval_good_X3,'nor') , wval_y_z3

wss3 = wX3[0]
wdf3 = pd.DataFrame(wss3)
for i in range (1,len(wX3)):
    wsss3 = wX3[i]
    wdf3 = pd.concat([wdf3, pd.DataFrame(wsss3)], axis=0)
#df = df.dropna(axis=0,how='any')
wvv3 = wval_X3[0]
wdf13 = pd.DataFrame(wvv3)
for j in range (1,len(wval_X3)):
    wvvv3 = wval_X3[j]
    wdf13 = pd.concat([wdf13, pd.DataFrame(wvvv3)], axis=0)
#   4
wgood_x14 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,5)
wgood_x24 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,5)
wgood_x34 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,5)
wgood_X4 = wgood_x14 + wgood_x24 + wgood_x34
wbad_x14 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,5)
wbad_x24 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,5)
wbad_x34 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,5)
wbad_X4 = wbad_x14 + wbad_x24 + wbad_x34
wval_good_x14 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,5)
wval_good_x24 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,5)
wval_good_x34 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,5)
wval_good_X4 = wval_good_x14 + wval_good_x24 + wval_good_x34
wval_bad_x14 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,5)
wval_bad_x24 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,5)
wval_bad_x34 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,5)
wval_bad_X4 = wval_bad_x14 + wval_bad_x24 + wval_bad_x34
wgood_y4 = np.zeros(len(wgood_X4))
wbad_y4  = np.ones(len(wbad_X4))
wtrain_y_z4 = np.concatenate([wgood_y4,wbad_y4])
wval_good_y4 = np.zeros([len(wval_good_X4)])
wval_bad_y4  = np.ones([len(wval_bad_X4)])
wval_y_z4 = np.concatenate([wval_good_y4,wval_bad_y4])
wgood_X4.extend(wbad_X4), wval_good_X4.extend(wval_bad_X4)
wX4, wy4, wval_X4, wval_y4 = nor(wgood_X4,'nor') , wtrain_y_z4 , nor(wval_good_X4,'nor') , wval_y_z4

wss4 = wX4[0]
wdf4 = pd.DataFrame(wss4)
for i in range (1,len(wX4)):
    wsss4 = wX4[i]
    wdf4 = pd.concat([wdf4, pd.DataFrame(wsss4)], axis=0)
#df = df.dropna(axis=0,how='any')
wvv4 = wval_X4[0]
wdf14 = pd.DataFrame(wvv4)
for j in range (1,len(wval_X4)):
    wvvv4 = wval_X4[j]
    wdf14 = pd.concat([wdf14, pd.DataFrame(wvvv4)], axis=0)
#   5
wgood_x15 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,5)
wgood_x25 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,5)
wgood_x35 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,5)
wgood_X5  = wgood_x15 + wgood_x25 + wgood_x35
wbad_x15 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,5)
wbad_x25 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,5)
wbad_x35 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,5)
wbad_X5 = wbad_x15 + wbad_x25 + wbad_x35
wval_good_x15 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,5)
wval_good_x25 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,5)
wval_good_x35 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,5)
wval_good_X5 = wval_good_x15 + wval_good_x25 + wval_good_x35
wval_bad_x15 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,5)
wval_bad_x25 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,5)
wval_bad_x35 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,5)
wval_bad_X5 = wval_bad_x15 + wval_bad_x25 + wval_bad_x35
wgood_y5 = np.zeros(len(wgood_X5))
wbad_y5  = np.ones(len(wbad_X5))
wtrain_y_z5 = np.concatenate([wgood_y5,wbad_y5])
wval_good_y5 = np.zeros([len(wval_good_X5)])
wval_bad_y5  = np.ones([len(wval_bad_X5)])
wval_y_z5 = np.concatenate([wval_good_y5,wval_bad_y5])
wgood_X5.extend(wbad_X5), wval_good_X5.extend(wval_bad_X5)
wX5, wy5, wval_X5, wval_y5 = nor(wgood_X5,'nor') , wtrain_y_z5 , nor(wval_good_X5,'nor') , wval_y_z5

wss5 = wX5[0]
wdf5 = pd.DataFrame(wss5)
for i in range (1,len(wX5)):
    wsss5 = wX5[i]
    wdf5 = pd.concat([wdf5, pd.DataFrame(wsss5)], axis=0)
#df = df.dropna(axis=0,how='any')
wvv5 = wval_X5[0]
wdf15 = pd.DataFrame(wvv5)
for j in range (1,len(wval_X5)):
    wvvv5 = wval_X5[j]
    wdf15 = pd.concat([wdf15, pd.DataFrame(wvvv5)], axis=0)
#   6
wgood_x16 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,5)
wgood_x26 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,5)
wgood_x36 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,5)
wgood_X6  = wgood_x16 + wgood_x26 + wgood_x36
wbad_x16 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,5)
wbad_x26 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,5)
wbad_x36 = frequency(path + 'all(WCST)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,5)
wbad_X6 = wbad_x16 + wbad_x26 + wbad_x36
wval_good_x16 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,5)
wval_good_x26 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,5)
wval_good_x36 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,5)
wval_good_X6 = wval_good_x16 + wval_good_x26 + wval_good_x36
wval_bad_x16 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,5)
wval_bad_x26 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,5)
wval_bad_x36 = frequency(path + 'all(WCST)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,5)
wval_bad_X6 = wval_bad_x16 + wval_bad_x26 + wval_bad_x36
wgood_y6 = np.zeros(len(wgood_X6))
wbad_y6  = np.ones(len(wbad_X6))
wtrain_y_z6 = np.concatenate([wgood_y6,wbad_y6])
wval_good_y6 = np.zeros([len(wval_good_X6)])
wval_bad_y6  = np.ones([len(wval_bad_X6)])
wval_y_z6 = np.concatenate([wval_good_y6,wval_bad_y6])
wgood_X6.extend(wbad_X6), wval_good_X6.extend(wval_bad_X6)
wX6, wy6, wval_X6, wval_y6 = nor(wgood_X6,'nor') , wtrain_y_z6 , nor(wval_good_X6,'nor') , wval_y_z6

wss6 = wX6[0]
wdf6 = pd.DataFrame(wss6)
for i in range (1,len(wX6)):
    wsss6 = wX6[i]
    wdf6 = pd.concat([wdf6, pd.DataFrame(wsss6)], axis=0)
#df = df.dropna(axis=0,how='any')
wvv6 = wval_X6[0]
wdf16 = pd.DataFrame(wvv6)
for j in range (1,len(wval_X6)):
    wvvv6 = wval_X6[j]
    wdf16 = pd.concat([wdf16, pd.DataFrame(wvvv6)], axis=0)
    
#VFT make
#    1
fgood_x11 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,3)
fgood_x21 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,3)
fgood_x31 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,3)
fgood_X1  = fgood_x11 + fgood_x21 + fgood_x31
fbad_x11 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,3)
fbad_x21 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,3)
fbad_x31 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,3)
fbad_X1 = fbad_x11 + fbad_x21 + fbad_x31
fval_good_x11 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,3)
fval_good_x21 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,3)
fval_good_x31 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,3)
fval_good_X1 = fval_good_x11 + fval_good_x21 + fval_good_x31
fval_bad_x11 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,3)
fval_bad_x21 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,3)
fval_bad_x31 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,3)
fval_bad_X1 = fval_bad_x11 + fval_bad_x21 + fval_bad_x31
fgood_y1 = np.zeros(len(fgood_X1))
fbad_y1  = np.ones(len(fbad_X1))
ftrain_y_z1 = np.concatenate([fgood_y1,fbad_y1])
fval_good_y1 = np.zeros([len(fval_good_X1)])
fval_bad_y1  = np.ones([len(fval_bad_X1)])
fval_y_z1 = np.concatenate([fval_good_y1,fval_bad_y1])
fgood_X1.extend(fbad_X1), fval_good_X1.extend(fval_bad_X1)
fX1, fy1, fval_X1, fval_y1 = nor(fgood_X1,'nor') , ftrain_y_z1 , nor(fval_good_X1,'nor') , fval_y_z1

fss1 = fX1[0]
fdf1 = pd.DataFrame(fss1)
for i in range (1,len(fX1)):
    fsss1 = fX1[i]
    fdf1 = pd.concat([fdf1, pd.DataFrame(fsss1)], axis=0)
#df = df.dropna(axis=0,how='any')
fvv1 = fval_X1[0]
fdf11 = pd.DataFrame(fvv1)
for j in range (1,len(fval_X1)):
    fvvv1 = fval_X1[j]
    fdf11 = pd.concat([fdf11, pd.DataFrame(fvvv1)], axis=0)
#    2
fgood_x12 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,3)
fgood_x22 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,3)
fgood_x32 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,3)
fgood_X2  = fgood_x12 + fgood_x22 + fgood_x32
fbad_x12 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,3)
fbad_x22 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,3)
fbad_x32 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,3)
fbad_X2 = fbad_x12 + fbad_x22 + fbad_x32
fval_good_x12 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,3)
fval_good_x22 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,3)
fval_good_x32 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,3)
fval_good_X2 = fval_good_x12 + fval_good_x22 + fval_good_x32
fval_bad_x12 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,3)
fval_bad_x22 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,3)
fval_bad_x32 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,3)
fval_bad_X2 = fval_bad_x12 + fval_bad_x22 + fval_bad_x32
fgood_y2 = np.zeros(len(fgood_X2))
fbad_y2  = np.ones(len(fbad_X2))
ftrain_y_z2 = np.concatenate([fgood_y2,fbad_y2])
fval_good_y2 = np.zeros([len(fval_good_X2)])
fval_bad_y2  = np.ones([len(fval_bad_X2)])
fval_y_z2 = np.concatenate([fval_good_y2,fval_bad_y2])
fgood_X2.extend(fbad_X2), fval_good_X2.extend(fval_bad_X2)
fX2, fy2, fval_X2, fval_y2 = nor(fgood_X2,'nor') , ftrain_y_z2 , nor(fval_good_X2,'nor') , fval_y_z2

fss2 = fX2[0]
fdf2 = pd.DataFrame(fss2)
for i in range (1,len(fX2)):
    fsss2 = fX2[i]
    fdf2 = pd.concat([fdf2, pd.DataFrame(fsss2)], axis=0)
#df = df.dropna(axis=0,how='any')
fvv2 = fval_X2[0]
fdf12 = pd.DataFrame(fvv2)
for j in range (1,len(fval_X2)):
    fvvv2 = fval_X2[j]
    fdf12 = pd.concat([fdf12, pd.DataFrame(fvvv2)], axis=0)
#    3
fgood_x13 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,3)
fgood_x23 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,3)
fgood_x33 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,3)
fgood_X3  = fgood_x13 + fgood_x23 + fgood_x33
fbad_x13 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,3)
fbad_x23 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,3)
fbad_x33 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,3)
fbad_X3 = fbad_x13 + fbad_x23 + fbad_x33
fval_good_x13 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,3)
fval_good_x23 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,3)
fval_good_x33 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,3)
fval_good_X3 = fval_good_x13 + fval_good_x23 + fval_good_x33
fval_bad_x13 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,3)
fval_bad_x23 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,3)
fval_bad_x33 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,3)
fval_bad_X3 = fval_bad_x13 + fval_bad_x23 + fval_bad_x33
fgood_y3 = np.zeros(len(fgood_X3))
fbad_y3  = np.ones(len(fbad_X3))
ftrain_y_z3 = np.concatenate([fgood_y3,fbad_y3])
fval_good_y3 = np.zeros([len(fval_good_X3)])
fval_bad_y3  = np.ones([len(fval_bad_X3)])
fval_y_z3 = np.concatenate([fval_good_y3,fval_bad_y3])
fgood_X3.extend(fbad_X3), fval_good_X3.extend(fval_bad_X3)
fX3, fy3, fval_X3, fval_y3 = nor(fgood_X3,'nor') , ftrain_y_z3 , nor(fval_good_X3,'nor') , fval_y_z3

fss3 = fX3[0]
fdf3 = pd.DataFrame(fss3)
for i in range (1,len(fX3)):
    fsss3 = fX3[i]
    fdf3 = pd.concat([fdf3, pd.DataFrame(fsss3)], axis=0)
#df = df.dropna(axis=0,how='any')
fvv3 = fval_X3[0]
fdf13 = pd.DataFrame(fvv3)
for j in range (1,len(fval_X3)):
    fvvv3 = fval_X3[j]
    fdf13 = pd.concat([fdf13, pd.DataFrame(fvvv3)], axis=0)
#    4
fgood_x1 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,3)
fgood_x2 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,3)
fgood_x3 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,3)
fgood_X  = fgood_x1 + fgood_x2 + fgood_x3
fbad_x1 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,3)
fbad_x2 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,3)
fbad_x3 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,3)
fbad_X = fbad_x1 + fbad_x2 + fbad_x3
fval_good_x1 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,3)
fval_good_x2 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,3)
fval_good_x3 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,3)
fval_good_X = fval_good_x1 + fval_good_x2 + fval_good_x3
fval_bad_x1 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,3)
fval_bad_x2 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,3)
fval_bad_x3 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,3)
fval_bad_X = fval_bad_x1 + fval_bad_x2 + fval_bad_x3
fgood_y = np.zeros(len(fgood_X))
fbad_y  = np.ones(len(fbad_X))
ftrain_y_z = np.concatenate([fgood_y,fbad_y])
fval_good_y = np.zeros([len(fval_good_X)])
fval_bad_y  = np.ones([len(fval_bad_X)])
fval_y_z = np.concatenate([fval_good_y,fval_bad_y])
fgood_X.extend(fbad_X), fval_good_X.extend(fval_bad_X)
fX, fy, fval_X, fval_y = nor(fgood_X,'nor') , ftrain_y_z , nor(fval_good_X,'nor') , fval_y_z

fss = fX[0]
fdf = pd.DataFrame(fss)
for i in range (1,len(fX)):
    fsss = fX[i]
    fdf = pd.concat([fdf, pd.DataFrame(fsss)], axis=0)
#df = df.dropna(axis=0,how='any')
fvv = fval_X[0]
fdf1 = pd.DataFrame(fvv)
for j in range (1,len(fval_X)):
    fvvv = fval_X[j]
    fdf1 = pd.concat([fdf1, pd.DataFrame(fvvv)], axis=0)
#   5
fgood_x1 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,3)
fgood_x2 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,3)
fgood_x3 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,3)
fgood_X  = fgood_x1 + fgood_x2 + fgood_x3
fbad_x1 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,3)
fbad_x2 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,3)
fbad_x3 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,3)
fbad_X = fbad_x1 + fbad_x2 + fbad_x3
fval_good_x1 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,3)
fval_good_x2 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,3)
fval_good_x3 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,3)
fval_good_X = fval_good_x1 + fval_good_x2 + fval_good_x3
fval_bad_x1 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,3)
fval_bad_x2 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,3)
fval_bad_x3 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,3)
fval_bad_X = fval_bad_x1 + fval_bad_x2 + fval_bad_x3
fgood_y = np.zeros(len(fgood_X))
fbad_y  = np.ones(len(fbad_X))
ftrain_y_z = np.concatenate([fgood_y,fbad_y])
fval_good_y = np.zeros([len(fval_good_X)])
fval_bad_y  = np.ones([len(fval_bad_X)])
fval_y_z = np.concatenate([fval_good_y,fval_bad_y])
fgood_X.extend(fbad_X), fval_good_X.extend(fval_bad_X)
fX, fy, fval_X, fval_y = nor(fgood_X,'nor') , ftrain_y_z , nor(fval_good_X,'nor') , fval_y_z

fss = fX[0]
fdf = pd.DataFrame(fss)
for i in range (1,len(fX)):
    fsss = fX[i]
    fdf = pd.concat([fdf, pd.DataFrame(fsss)], axis=0)
#df = df.dropna(axis=0,how='any')
fvv = fval_X[0]
fdf1 = pd.DataFrame(fvv)
for j in range (1,len(fval_X)):
    fvvv = fval_X[j]
    fdf1 = pd.concat([fdf1, pd.DataFrame(fvvv)], axis=0)
#    6
fgood_x1 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 2, 20,1119,3)
fgood_x2 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 4, 25,1119,3)
fgood_x3 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[0] + '\\', 6, 30,1119,3)
fgood_X  = fgood_x1 + fgood_x2 + fgood_x3
fbad_x1 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 2, 20,1119,3)
fbad_x2 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 4, 25,1119,3)
fbad_x3 = frequency(path + 'all(VFT)\\ch1_16\\train\\' + class_name[1] + '\\', 6, 30,1119,3)
fbad_X = fbad_x1 + fbad_x2 + fbad_x3
fval_good_x1 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 2, 20,1119,3)
fval_good_x2 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 4, 30,1119,3)
fval_good_x3 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[0] + '\\', 6, 30,1119,3)
fval_good_X = fval_good_x1 + fval_good_x2 + fval_good_x3
fval_bad_x1 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 2, 30,1119,3)
fval_bad_x2 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 4, 30,1119,3)
fval_bad_x3 = frequency(path + 'all(VFT)\\ch1_16\\val\\' + class_name[1] + '\\', 6, 30,1119,3)
fval_bad_X = fval_bad_x1 + fval_bad_x2 + fval_bad_x3
fgood_y = np.zeros(len(fgood_X))
fbad_y  = np.ones(len(fbad_X))
ftrain_y_z = np.concatenate([fgood_y,fbad_y])
fval_good_y = np.zeros([len(fval_good_X)])
fval_bad_y  = np.ones([len(fval_bad_X)])
fval_y_z = np.concatenate([fval_good_y,fval_bad_y])
fgood_X.extend(fbad_X), fval_good_X.extend(fval_bad_X)
fX, fy, fval_X, fval_y = nor(fgood_X,'nor') , ftrain_y_z , nor(fval_good_X,'nor') , fval_y_z

fss = fX[0]
fdf = pd.DataFrame(fss)
for i in range (1,len(fX)):
    fsss = fX[i]
    fdf = pd.concat([fdf, pd.DataFrame(fsss)], axis=0)
#df = df.dropna(axis=0,how='any')
fvv = fval_X[0]
fdf1 = pd.DataFrame(fvv)
for j in range (1,len(fval_X)):
    fvvv = fval_X[j]
    fdf1 = pd.concat([fdf1, pd.DataFrame(fvvv)], axis=0)
    
#TMTA
agood_x1 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[0] + '\\A\\', 2, 20,1600,5)
agood_x2 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[0] + '\\A\\', 4, 25,1600,5)
agood_x3 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[0] + '\\A\\', 6, 30,1600,5)
agood_X  = agood_x1 + agood_x2 + agood_x3
abad_x1 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[1] + '\\A\\', 2, 20,1600,5)
abad_x2 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[1] + '\\A\\', 4, 25,1600,5)
abad_x3 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[1] + '\\A\\', 6, 30,1600,5)
abad_X = abad_x1 + abad_x2 + abad_x3
aval_good_x1 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[0] + '\\A\\', 2, 20,1600,5)
aval_good_x2 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[0] + '\\A\\', 4, 30,1600,5)
aval_good_x3 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[0] + '\\A\\', 6, 30,1600,5)
aval_good_X = aval_good_x1 + aval_good_x2 + aval_good_x3
aval_bad_x1 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[1] + '\\A\\', 2, 30,1600,5)
aval_bad_x2 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[1] + '\\A\\', 4, 30,1600,5)
aval_bad_x3 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[1] + '\\A\\', 6, 30,1600,5)
aval_bad_X = aval_bad_x1 + aval_bad_x2 + aval_bad_x3
agood_y = np.zeros(len(agood_X))
abad_y  = np.ones(len(abad_X))
atrain_y_z = np.concatenate([agood_y,abad_y])
aval_good_y = np.zeros([len(aval_good_X)])
aval_bad_y  = np.ones([len(aval_bad_X)])
aval_y_z = np.concatenate([aval_good_y,aval_bad_y])
agood_X.extend(abad_X), aval_good_X.extend(aval_bad_X)
aX, ay, aval_X, aval_y = nor(agood_X,'nor') , atrain_y_z , nor(aval_good_X,'nor') , aval_y_z

ass = aX[0]
adf = pd.DataFrame(ass)
for i in range (1,len(aX)):
    asss = aX[i]
    adf = pd.concat([adf, pd.DataFrame(asss)], axis=0)
#df = df.dropna(axis=0,how='any')
avv = aval_X[0]
adf1 = pd.DataFrame(avv)
for j in range (1,len(aval_X)):
    avvv = aval_X[j]
    adf1 = pd.concat([adf1, pd.DataFrame(avvv)], axis=0)

#TMTB
bgood_x1 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[0] + '\\B\\', 2, 20,3600,1)
bgood_x2 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[0] + '\\B\\', 4, 25,3600,1)
bgood_x3 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[0] + '\\B\\', 6, 30,3600,1)
bgood_X  = bgood_x1 + bgood_x2 + bgood_x3
bbad_x1 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[1] + '\\B\\', 2, 20,3600,1)
bbad_x2 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[1] + '\\B\\', 4, 25,3600,1)
bbad_x3 = frequency(path + 'all(TMT)v\\ch1_16\\train\\' + class_name[1] + '\\B\\', 6, 30,3600,1)
bbad_X = bbad_x1 + bbad_x2 + bbad_x3
bval_good_x1 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[0] + '\\B\\', 2, 20,3600,1)
bval_good_x2 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[0] + '\\B\\', 4, 30,3600,1)
bval_good_x3 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[0] + '\\B\\', 6, 30,3600,1)
bval_good_X = bval_good_x1 + bval_good_x2 + bval_good_x3
bval_bad_x1 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[1] + '\\B\\', 2, 30,3600,1)
bval_bad_x2 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[1] + '\\B\\', 4, 30,3600,1)
bval_bad_x3 = frequency(path + 'all(TMT)v\\ch1_16\\val\\' + class_name[1] + '\\B\\', 6, 30,3600,1)
bval_bad_X = bval_bad_x1 + bval_bad_x2 + bval_bad_x3
bgood_y = np.zeros(len(bgood_X))
bbad_y  = np.ones(len(bbad_X))
btrain_y_z = np.concatenate([bgood_y,bbad_y])
bval_good_y = np.zeros([len(bval_good_X)])
bval_bad_y  = np.ones([len(bval_bad_X)])
bval_y_z = np.concatenate([bval_good_y,bval_bad_y])
bgood_X.extend(bbad_X), bval_good_X.extend(bval_bad_X)
bX, by, bval_X, bval_y = nor(bgood_X,'nor') , btrain_y_z , nor(bval_good_X,'nor') , bval_y_z

bss = bX[0]
bdf = pd.DataFrame(bss)
for i in range (1,len(bX)):
    bsss = bX[i]
    bdf = pd.concat([bdf, pd.DataFrame(bsss)], axis=0)
#df = df.dropna(axis=0,how='any')
bvv = bval_X[0]
bdf1 = pd.DataFrame(bvv)
for j in range (1,len(bval_X)):
    bvvv = bval_X[j]
    bdf1 = pd.concat([bdf1, pd.DataFrame(bvvv)], axis=0)

for i  in range (3,11): 
    print('max_depth='+str(i))
#    df1 = df1.dropna(axis=0,how='any')
    wmy_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    fmy_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    amy_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    bmy_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    # Add silent=True to avoid printing out updates with each cycle
    wmy_model.fit(wdf, wy, verbose=False)
    fmy_model.fit(fdf, fy, verbose=False)
    amy_model.fit(adf, ay, verbose=False)
    bmy_model.fit(bdf, by, verbose=False)
#read & pre
#   WCST
    wgx = read_data(path + 'all(WCST)\\ch1_16\\test\\' + class_name[0] + '\\',5 )
    wbx = read_data(path + 'all(WCST)\\ch1_16\\test\\' + class_name[1] + '\\',5 )
    wgood_y = np.zeros([len(wgx)])
    wbad_y  = np.ones([len(wbx)])
    wtest_y_z = np.concatenate([wgood_y,wbad_y])
    wgx.extend(wbx)
    wtest_X, wtest_y = nor(wgx,'nor') , wtest_y_z
    wtt = wtest_X[0]
    wdf2 = pd.DataFrame(wtt)
    for k in range (1,len(wtest_X)):
        wttt = wtest_X[k]
        wdf2 = pd.concat([wdf2, pd.DataFrame(wttt)], axis=0)
    wdf2 = wdf2.dropna(axis=0,how='any')   
    wval_y_pred2 = wmy_model.predict(wdf2)
    
#   VFT
    fgx = read_data(path + 'all(VFT)\\ch1_16\\test\\' + class_name[0] + '\\',3 )
    fbx = read_data(path + 'all(VFT)\\ch1_16\\test\\' + class_name[1] + '\\',3 )
    fgood_y = np.zeros([len(fgx)])
    fbad_y  = np.ones([len(fbx)])
    ftest_y_z = np.concatenate([fgood_y,fbad_y])
    fgx.extend(fbx)
    ftest_X, ftest_y = nor(fgx,'nor') , ftest_y_z
    ftt = ftest_X[0]
    fdf2 = pd.DataFrame(ftt)
    for k in range (1,len(ftest_X)):
        fttt = ftest_X[k]
        fdf2 = pd.concat([fdf2, pd.DataFrame(fttt)], axis=0)
    fdf2 = fdf2.dropna(axis=0,how='any')   
    fval_y_pred2 = fmy_model.predict(fdf2)
    
#   TMTA
    agx = read_data(path + 'all(TMT)v\\ch1_16\\test\\' + class_name[0] + '\\A\\',5 )
    abx = read_data(path + 'all(TMT)v\\ch1_16\\test\\' + class_name[1] + '\\A\\',5 )
    agood_y = np.zeros([len(agx)])
    abad_y  = np.ones([len(abx)])
    atest_y_z = np.concatenate([agood_y,abad_y])
    agx.extend(abx)
    atest_X, atest_y = nor(agx,'nor') , atest_y_z
    att = atest_X[0]
    adf2 = pd.DataFrame(att)
    for k in range (1,len(atest_X)):
        attt = atest_X[k]
        adf2 = pd.concat([adf2, pd.DataFrame(attt)], axis=0)
    adf2 = adf2.dropna(axis=0,how='any')   
    aval_y_pred2 = amy_model.predict(adf2)
    
#   TMTB
    bgx = read_data(path + 'all(TMT)v\\ch1_16\\test\\' + class_name[0] + '\\B\\',1 )
    bbx = read_data(path + 'all(TMT)v\\ch1_16\\test\\' + class_name[1] + '\\B\\',1 )
    bgood_y = np.zeros([len(bgx)])
    bbad_y  = np.ones([len(bbx)])
    btest_y_z = np.concatenate([bgood_y,bbad_y])
    bgx.extend(bbx)
    btest_X, btest_y = nor(bgx,'nor') , btest_y_z
    btt = btest_X[0]
    bdf2 = pd.DataFrame(btt)
    for k in range (1,len(btest_X)):
        bttt = btest_X[k]
        bdf2 = pd.concat([bdf2, pd.DataFrame(bttt)], axis=0)
    bdf2 = bdf2.dropna(axis=0,how='any')   
    bval_y_pred2 = bmy_model.predict(bdf2)
    
#   ensemble
    plal = wval_y_pred2 + fval_y_pred2 + aval_y_pred2 + bval_y_pred2
    l = plal.shape
    for i in range (0,l[0]):
        if plal[i] >= 2:
            plal[i] = 1
        else:
            plal[i] = 0

#   ACC & AUC
    auct = roc_auc_score(atest_y, plal)
    print("Performance sur le test auc : ", auct)
    acct = accuracy_score(atest_y, plal)
    print("Performance sur le test acc : ", acct)
    kk = atest_y - plal
    print(np.where(kk != 0))