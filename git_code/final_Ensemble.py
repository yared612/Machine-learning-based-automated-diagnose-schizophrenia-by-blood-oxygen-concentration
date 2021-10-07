from sklearn.metrics import roc_auc_score
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
from XGBensemble_utils import *

train_file = '/media/yared/SP PHD U3/研究所/HCSZ/ch1_16/train'
val_file   = '//media/yared/SP PHD U3/研究所/HCSZ/ch1_16/val'
path = '/media/yared/SP PHD U3/研究所/HCSZ/'

#--------------------------------------------------catch category--------------------------------------------------
class_ = glob.glob(train_file + '/*')
class_name = []
for name in class_:
    class_name.append(name.split('/')[-1])
#--------------------------------------------------catch category End--------------------------------------------------    

#--------------------------------------------------function--------------------------------------------------
#--------------------內插--------------------
def data_aug1(data_file,mm,cc,crop_number = 10):
    filename = glob.glob(data_file + '*.csv')
    im_name = []
    for i in filename:
        n = i.split('/')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    for file in im_name:
        data = pd.read_csv(data_file + file + '.csv')
        data = np.array(data)[:,cc]
        data = np.reshape(data,data.shape+(1,))
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

def data_aug2(data_file,cc,crop_number = 10):
    filename = glob.glob(data_file + 'A/' + '*.csv')
    im_name = []
    for i in filename:
        n = i.split('/')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    for file in im_name:
        data = pd.read_csv(data_file + 'A/' + file + '.csv')
        data1 = pd.read_csv(data_file + 'B/' + file + '.csv')
        data = np.array(data)[:,0]
        data1 = np.array(data1)[:,0]
        data = np.reshape(data,data.shape+(1,))
        data1 = np.reshape(data1,data1.shape+(1,))
        data_crop  = []  
        data_cropb  = []
        
        for chan in range(0,data.shape[1]):
            for num in range(0,len(data),crop_number):
                data_crop1 = data[num:num+crop_number,chan]
                random     = choice(data_crop1)
                data_crop.append(random)
        for chan1 in range(0,data1.shape[1]):
            for num1 in range(0,len(data1),crop_number):
                data_crop12 = data1[num1:num1+crop_number,chan1]
                random1     = choice(data_crop12)
                data_cropb.append(random1)           
        c =[]
        d = []
        
        k = math.ceil(len(data_crop)/data.shape[1])
        for i in range(0,len(data_crop),k):
            c0 = data_crop[i:i+k]
            x0 = np.linspace(0,len(c0)-1,len(c0))
            x1 = np.linspace(0,len(c0)-1,1570)
            y0 = np.interp(x1,x0,c0)
            c.append(y0)            
        c = np.stack(c,axis=-1)
        
        k1 = math.ceil(len(data_cropb)/data1.shape[1])
        for i in range(0,len(data_cropb),k1):
            c0b = data_crop[i:i+k1]
            x0b = np.linspace(0,len(c0b)-1,len(c0b))
            x1b = np.linspace(0,len(c0b)-1,3600)
            y0b = np.interp(x1b,x0b,c0b)
            d.append(y0b)            
        d = np.stack(d,axis=-1)
        c = np.concatenate((c, d), axis=0)
        data_split_out.append(c.T)
    return data_split_out

def data_aug3(data_file,mm,cc,crop_number = 10):
    filename = glob.glob(data_file + '*.csv')
    im_name = []
    for i in filename:
        n = i.split('/')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    for file in im_name:
        data = pd.read_csv(data_file + file + '.csv')
        pg = pd.read_csv('/home/yared/文件/HCSZ/all(only WCST grade)/ch1_16/' + data_file.split('/')[-3] + '/' + 
                         data_file.split('/')[-2] + '/' + file + '.csv')
        data = np.array(data)[:,cc]
        pg  = np.array(pg)[:,1]
        data = np.reshape(data,data.shape+(1,))
        pg = np.reshape(pg,pg.shape+(1,))
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
        c = np.concatenate((c, pg), axis=0)
        data_split_out.append(c.T)
    return data_split_out
#--------------------內插 End--------------------
    
#--------------------讀圖頻率(aug用)--------------------
def frequency(data_file,num,crop_number,mm,cc):
    final = []
    for i in range(1,num):
        x = data_aug1(data_file,mm,cc,crop_number)
        final+=x     
    return final

def frequency1(data_file,num,crop_number,cc):
    final = []
    for i in range(1,num):
        x = data_aug2(data_file,cc,crop_number)
        final+=x     
    return final

def frequency2(data_file,num,crop_number,mm,cc):
    final = []
    for i in range(1,num):
        x = data_aug3(data_file,mm,cc,crop_number)
        final+=x     
    return final
#--------------------讀圖頻率(aug用) End--------------------
    
#--------------------pre-processing--------------------
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
#--------------------pre-processing End--------------------
    
#--------------------read test data--------------------
def read_data(data_file,cc):
    filename = glob.glob(data_file +'/*.csv')
    im_name = []
    for i in filename:
        n = i.split('/')[-1].split('.')[0]
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

def read_data1(data_file,cc):
    filename = glob.glob(data_file +'/*.csv')
    im_name = []
    for i in filename:
        n = i.split('/')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    aa = []
    for file in im_name:
        data = pd.read_csv(data_file + file + '.csv')
        pg = pd.read_csv('/home/yared/文件/HCSZ/all(only WCST grade)/ch1_16/' + data_file.split('/')[-3] + '/' + 
                         data_file.split('/')[-2] + '/' + file + '.csv')
        data = np.array(data)[:,cc]
        pg  = np.array(pg)[:,1]
        pg = np.reshape(pg,pg.shape+(1,))
        data = np.reshape(data,data.shape+(1,))
        data = np.concatenate((data, pg), axis=0)
        aa.append(data.T)            
    for k in range(0,len(aa)):
        data_split_out.append(aa[k])
    return data_split_out
#--------------------read test data End--------------------
    
#--------------------read train、 val data & augmentation--------------------
#WCST & VFT
def preprowf(path,it,mm,cc):
    wgood_x11 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/', 2, 20,mm,cc)
    wgood_x21 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/', 4, 25,mm,cc)
    wgood_x31 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/', 6, 30,mm,cc)
    wgood_X1  = wgood_x11 + wgood_x21 + wgood_x31
    wbad_x11 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/', 2, 20,mm,cc)
    wbad_x21 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/', 4, 25,mm,cc)
    wbad_x31 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/', 6, 30,mm,cc)
    wbad_X1 = wbad_x11 + wbad_x21 + wbad_x31
    wval_good_x11 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/', 2, 20,mm,cc)
    wval_good_x21 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/', 4, 30,mm,cc)
    wval_good_x31 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/', 6, 30,mm,cc)
    wval_good_X1 = wval_good_x11 + wval_good_x21 + wval_good_x31
    wval_bad_x11 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/', 2, 30,mm,cc)
    wval_bad_x21 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/', 4, 30,mm,cc)
    wval_bad_x31 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/', 6, 30,mm,cc)
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
    return wdf1, wy1, wdf11, wval_y1

#TMT_A(plus time)
def preproa(path,it,mm,cc):
    wgood_x11 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/A/', 2, 20,mm,cc)
    wgood_x21 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/A/', 4, 25,mm,cc)
    wgood_x31 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/A/', 6, 30,mm,cc)
    wgood_X1  = wgood_x11 + wgood_x21 + wgood_x31
    wbad_x11 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/A/', 2, 20,mm,cc)
    wbad_x21 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/A/', 4, 25,mm,cc)
    wbad_x31 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/A/', 6, 30,mm,cc)
    wbad_X1 = wbad_x11 + wbad_x21 + wbad_x31
    wval_good_x11 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/A/', 2, 20,mm,cc)
    wval_good_x21 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/A/', 4, 30,mm,cc)
    wval_good_x31 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/A/', 6, 30,mm,cc)
    wval_good_X1 = wval_good_x11 + wval_good_x21 + wval_good_x31
    wval_bad_x11 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/A/', 2, 30,mm,cc)
    wval_bad_x21 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/A/', 4, 30,mm,cc)
    wval_bad_x31 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/A/', 6, 30,mm,cc)
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
    return wdf1, wy1, wdf11, wval_y1

#TMT_A(ori)
def preprooria(path,it,mm,cc):
    wgood_x11 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/A/', 2, 2,mm,cc)
    wgood_x21 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/A/', 4, 4,mm,cc)
    wgood_x31 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/A/', 6, 6,mm,cc)
    wgood_X1  = wgood_x11 + wgood_x21 + wgood_x31
    wbad_x11 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/A/', 2, 2,mm,cc)
    wbad_x21 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/A/', 4, 4,mm,cc)
    wbad_x31 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/A/', 6, 6,mm,cc)
    wbad_X1 = wbad_x11 + wbad_x21 + wbad_x31
    wval_good_x11 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/A/', 2, 2,mm,cc)
    wval_good_x21 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/A/', 4, 4,mm,cc)
    wval_good_x31 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/A/', 6, 6,mm,cc)
    wval_good_X1 = wval_good_x11 + wval_good_x21 + wval_good_x31
    wval_bad_x11 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/A/', 2, 2,mm,cc)
    wval_bad_x21 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/A/', 4, 4,mm,cc)
    wval_bad_x31 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/A/', 6, 6,mm,cc)
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
    return wdf1, wy1, wdf11, wval_y1

#TMTB(plus time)
def preprob(path,it,mm,cc):
    wgood_x11 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/B/', 2, 20,mm,cc)
    wgood_x21 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/B/', 4, 25,mm,cc)
    wgood_x31 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/B/', 6, 30,mm,cc)
    wgood_X1  = wgood_x11 + wgood_x21 + wgood_x31
    wbad_x11 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/B/', 2, 20,mm,cc)
    wbad_x21 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/B/', 4, 25,mm,cc)
    wbad_x31 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/B/', 6, 30,mm,cc)
    wbad_X1 = wbad_x11 + wbad_x21 + wbad_x31
    wval_good_x11 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/B/', 2, 20,mm,cc)
    wval_good_x21 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/B/', 4, 30,mm,cc)
    wval_good_x31 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/B/', 6, 30,mm,cc)
    wval_good_X1 = wval_good_x11 + wval_good_x21 + wval_good_x31
    wval_bad_x11 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/B/', 2, 30,mm,cc)
    wval_bad_x21 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/B/', 4, 30,mm,cc)
    wval_bad_x31 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/B/', 6, 30,mm,cc)
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
    return wdf1, wy1, wdf11, wval_y1

#TMT_B(ori)
def preproorib(path,it,mm,cc):
    wgood_x11 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/B/', 2, 2,mm,cc)
    wgood_x21 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/B/', 4, 4,mm,cc)
    wgood_x31 = frequency(path + it + '/ch1_16/train/' + class_name[0] + '/B/', 6, 6,mm,cc)
    wgood_X1  = wgood_x11 + wgood_x21 + wgood_x31
    wbad_x11 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/B/', 2, 2,mm,cc)
    wbad_x21 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/B/', 4, 4,mm,cc)
    wbad_x31 = frequency(path + it + '/ch1_16/train/' + class_name[1] + '/B/', 6, 6,mm,cc)
    wbad_X1 = wbad_x11 + wbad_x21 + wbad_x31
    wval_good_x11 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/B/', 2, 2,mm,cc)
    wval_good_x21 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/B/', 4, 4,mm,cc)
    wval_good_x31 = frequency(path + it + '/ch1_16/val/' + class_name[0] + '/B/', 6, 6,mm,cc)
    wval_good_X1 = wval_good_x11 + wval_good_x21 + wval_good_x31
    wval_bad_x11 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/B/', 2, 2,mm,cc)
    wval_bad_x21 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/B/', 4, 4,mm,cc)
    wval_bad_x31 = frequency(path + it + '/ch1_16/val/' + class_name[1] + '/B/', 6, 6,mm,cc)
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
    return wdf1, wy1, wdf11, wval_y1

#TMT_A+TMT_B (ori)
def preprooriab(path,it,cc):
    wgood_x11 = frequency1(path + it + '/ch1_16/train/' + class_name[0] + '/', 2, 20,cc)
    wgood_x21 = frequency1(path + it + '/ch1_16/train/' + class_name[0] + '/', 4, 25,cc)
    wgood_x31 = frequency1(path + it + '/ch1_16/train/' + class_name[0] + '/', 6, 30,cc)
    wgood_X1  = wgood_x11 + wgood_x21 + wgood_x31
    wbad_x11 = frequency1(path + it + '/ch1_16/train/' + class_name[1] + '/', 2, 20,cc)
    wbad_x21 = frequency1(path + it + '/ch1_16/train/' + class_name[1] + '/', 4, 25,cc)
    wbad_x31 = frequency1(path + it + '/ch1_16/train/' + class_name[1] + '/', 6, 30,cc)
    wbad_X1 = wbad_x11 + wbad_x21 + wbad_x31
    wval_good_x11 = frequency1(path + it + '/ch1_16/val/' + class_name[0] + '/', 2, 20,cc)
    wval_good_x21 = frequency1(path + it + '/ch1_16/val/' + class_name[0] + '/', 4, 30,cc)
    wval_good_x31 = frequency1(path + it + '/ch1_16/val/' + class_name[0] + '/', 6, 30,cc)
    wval_good_X1 = wval_good_x11 + wval_good_x21 + wval_good_x31
    wval_bad_x11 = frequency1(path + it + '/ch1_16/val/' + class_name[1] + '/', 2, 30,cc)
    wval_bad_x21 = frequency1(path + it + '/ch1_16/val/' + class_name[1] + '/', 4, 30,cc)
    wval_bad_x31 = frequency1(path + it + '/ch1_16/val/' + class_name[1] + '/', 6, 30,cc)
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
    return wdf1, wy1, wdf11, wval_y1

#WCST+grade
def preprowpg(path,it,mm,cc):
    wgood_x11 = frequency2(path + it + '/ch1_16/train/' + class_name[0] + '/', 2, 20,mm,cc)
    wgood_x21 = frequency2(path + it + '/ch1_16/train/' + class_name[0] + '/', 4, 25,mm,cc)
    wgood_x31 = frequency2(path + it + '/ch1_16/train/' + class_name[0] + '/', 6, 30,mm,cc)
    wgood_X1  = wgood_x11 + wgood_x21 + wgood_x31
    wbad_x11 = frequency2(path + it + '/ch1_16/train/' + class_name[1] + '/', 2, 20,mm,cc)
    wbad_x21 = frequency2(path + it + '/ch1_16/train/' + class_name[1] + '/', 4, 25,mm,cc)
    wbad_x31 = frequency2(path + it + '/ch1_16/train/' + class_name[1] + '/', 6, 30,mm,cc)
    wbad_X1 = wbad_x11 + wbad_x21 + wbad_x31
    wval_good_x11 = frequency2(path + it + '/ch1_16/val/' + class_name[0] + '/', 2, 20,mm,cc)
    wval_good_x21 = frequency2(path + it + '/ch1_16/val/' + class_name[0] + '/', 4, 30,mm,cc)
    wval_good_x31 = frequency2(path + it + '/ch1_16/val/' + class_name[0] + '/', 6, 30,mm,cc)
    wval_good_X1 = wval_good_x11 + wval_good_x21 + wval_good_x31
    wval_bad_x11 = frequency2(path + it + '/ch1_16/val/' + class_name[1] + '/', 2, 30,mm,cc)
    wval_bad_x21 = frequency2(path + it + '/ch1_16/val/' + class_name[1] + '/', 4, 30,mm,cc)
    wval_bad_x31 = frequency2(path + it + '/ch1_16/val/' + class_name[1] + '/', 6, 30,mm,cc)
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
    return wdf1, wy1, wdf11, wval_y1
#--------------------read train、 val data & augmentation End--------------------
    
#--------------------predict--------------------
#WCST & VFT
def pred(path,X,y,it,cc):
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    my_model.fit(X, y, verbose=False)
    gx = read_data(path + it + '/ch1_16/test/' + class_name[0] + '/',cc )
    bx = read_data(path + it + '/ch1_16/test/' + class_name[1] + '/',cc )
    good_y = np.zeros([len(gx)])
    bad_y  = np.ones([len(bx)])
    test_y_z = np.concatenate([good_y,bad_y])
    gx.extend(bx)
    test_X, test_y = nor(gx,'nor') , test_y_z
    tt = test_X[0]
    df2 = pd.DataFrame(tt)
    for k in range (1,len(test_X)):
        ttt = test_X[k]
        df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
    df2 = df2.dropna(axis=0,how='any')   
    val_y_pred2 = my_model.predict(df2)
    return val_y_pred2, test_y

#TMT_A(plus time)
def preda(path,X,y,it,cc):
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    my_model.fit(X, y, verbose=False)
    gx = read_data(path + it + '/ch1_16/test/' + class_name[0] + '/A/',cc )
    bx = read_data(path + it + '/ch1_16/test/' + class_name[1] + '/A/',cc )
    good_y = np.zeros([len(gx)])
    bad_y  = np.ones([len(bx)])
    test_y_z = np.concatenate([good_y,bad_y])
    gx.extend(bx)
    test_X, test_y = nor(gx,'nor') , test_y_z
    tt = test_X[0]
    df2 = pd.DataFrame(tt)
    for k in range (1,len(test_X)):
        ttt = test_X[k]
        df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
    df2 = df2.dropna(axis=0,how='any')   
    val_y_pred2 = my_model.predict(df2)
    return val_y_pred2, test_y

#TMT_B(plus time)
def predb(path,X,y,it,cc):
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    my_model.fit(X, y, verbose=False)
    gx = read_data(path + it + '/ch1_16/test/' + class_name[0] + '/B/',cc )
    bx = read_data(path + it + '/ch1_16/test/' + class_name[1] + '/B/',cc )
    good_y = np.zeros([len(gx)])
    bad_y  = np.ones([len(bx)])
    test_y_z = np.concatenate([good_y,bad_y])
    gx.extend(bx)
    test_X, test_y = nor(gx,'nor') , test_y_z
    tt = test_X[0]
    df2 = pd.DataFrame(tt)
    for k in range (1,len(test_X)):
        ttt = test_X[k]
        df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
    df2 = df2.dropna(axis=0,how='any')   
    val_y_pred2 = my_model.predict(df2)
    return val_y_pred2, test_y

#TMT_A(ori)
def predoria(path,X,y,it,cc):
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    my_model.fit(X, y, verbose=False)
    gx = frequency(path + it + '/ch1_16/test/' + class_name[0] + '/A/', 2, 2,1570,cc )
    bx = frequency(path + it + '/ch1_16/test/' + class_name[1] + '/A/', 2, 2,1570,cc )
    good_y = np.zeros([len(gx)])
    bad_y  = np.ones([len(bx)])
    test_y_z = np.concatenate([good_y,bad_y])
    gx.extend(bx)
    test_X, test_y = nor(gx,'nor') , test_y_z
    tt = test_X[0]
    df2 = pd.DataFrame(tt)
    for k in range (1,len(test_X)):
        ttt = test_X[k]
        df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
    df2 = df2.dropna(axis=0,how='any')   
    val_y_pred2 = my_model.predict(df2)
    return val_y_pred2, test_y

#TMT_B(ori)
def predorib(path,X,y,it,cc):
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    my_model.fit(X, y, verbose=False)
    gx = frequency(path + it + '/ch1_16/test/' + class_name[0] + '/B/', 2, 2,3600,cc )
    bx = frequency(path + it + '/ch1_16/test/' + class_name[1] + '/B/', 2, 2,3600,cc )
    good_y = np.zeros([len(gx)])
    bad_y  = np.ones([len(bx)])
    test_y_z = np.concatenate([good_y,bad_y])
    gx.extend(bx)
    test_X, test_y = nor(gx,'nor') , test_y_z
    tt = test_X[0]
    df2 = pd.DataFrame(tt)
    for k in range (1,len(test_X)):
        ttt = test_X[k]
        df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
    df2 = df2.dropna(axis=0,how='any')   
    val_y_pred2 = my_model.predict(df2)
    return val_y_pred2, test_y

#TMT_A+TMT_B (ori)
def predoriab(path,X,y,it,cc):
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    my_model.fit(X, y, verbose=False)
    gx = frequency1(path + it + '/ch1_16/test/' + class_name[0] + '/', 2, 2,cc )
    bx = frequency1(path + it + '/ch1_16/test/' + class_name[1] + '/', 2, 2,cc )
    good_y = np.zeros([len(gx)])
    bad_y  = np.ones([len(bx)])
    test_y_z = np.concatenate([good_y,bad_y])
    gx.extend(bx)
    test_X, test_y = nor(gx,'nor') , test_y_z
    tt = test_X[0]
    df2 = pd.DataFrame(tt)
    for k in range (1,len(test_X)):
        ttt = test_X[k]
        df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
    df2 = df2.dropna(axis=0,how='any')   
    val_y_pred2 = my_model.predict(df2)
    return val_y_pred2, test_y

#WCST+grade
def predwpg(path,X,y,it,cc):
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    my_model.fit(X, y, verbose=False)
    gx = read_data1(path + it + '/ch1_16/test/' + class_name[0] + '/',cc )
    bx = read_data1(path + it + '/ch1_16/test/' + class_name[1] + '/',cc )
    good_y = np.zeros([len(gx)])
    bad_y  = np.ones([len(bx)])
    test_y_z = np.concatenate([good_y,bad_y])
    gx.extend(bx)
    test_X, test_y = nor(gx,'nor') , test_y_z
    tt = test_X[0]
    df2 = pd.DataFrame(tt)
    for k in range (1,len(test_X)):
        ttt = test_X[k]
        df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
    df2 = df2.dropna(axis=0,how='any')   
    val_y_pred2 = my_model.predict(df2)
    return val_y_pred2, test_y

#new(WCST)
def prednew(path,X,y,it,cc):
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    my_model.fit(X, y, verbose=False)
    gx = frequency(path + it + '/ch1_16/test/' + class_name[0] + '/' , 2, 2,1119,cc )
    bx = frequency(path + it + '/ch1_16/test/' + class_name[1] + '/' , 2, 2,1119,cc )
    good_y = np.zeros([len(gx)])
    bad_y  = np.ones([len(bx)])
    test_y_z = np.concatenate([good_y,bad_y])
    gx.extend(bx)
    test_X, test_y = nor(gx,'nor') , test_y_z
    tt = test_X[0]
    df2 = pd.DataFrame(tt)
    for k in range (1,len(test_X)):
        ttt = test_X[k]
        df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
    df2 = df2.dropna(axis=0,how='any')   
    val_y_pred2 = my_model.predict(df2)
    return val_y_pred2, test_y
#--------------------predict End--------------------
#--------------------------------------------------function End--------------------------------------------------
        
#--------------------------------------------------main code--------------------------------------------------
#--------------------make train and val data--------------------
#WCST 
#wX1, wy1, WvX1, wvy1 = preprowf(path,'all(WCST)',1119,9)
#wX2, wy2, WvX2, wvy2 = preprowf(path,'all(WCST)',1119,14)
#wX3, wy3, WvX3, wvy3 = preprowf(path,'all(WCST)',1119,15)
#wX4, wy4, WvX4, wvy4 = preprowf(path,'all(WCST)',1119,9)
#wX5, wy5, WvX5, wvy5 = preprowf(path,'all(WCST)',1119,11)
#wX6, wy6, WvX6, wvy6 = preprowf(path,'all(WCST)',1119,11)
#wX1, wy1, WvX1, wvy1 = preprowf(path,'WCST(50-1)',1119,0)    

##VFT 
fX1, fy1, fvX1, fvy1 = preprowf(path,'all(VFT)',1119,5)
fX2, fy2, fvX2, fvy2 = preprowf(path,'all(VFT)',1119,6)
fX3, fy3, fvX3, fvy3 = preprowf(path,'all(VFT)',1119,8)
fX4, fy4, fvX4, fvy4 = preprowf(path,'all(VFT)',1119,9)
fX5, fy5, fvX5, fvy5 = preprowf(path,'all(VFT)',1119,11)
#fX6, fy6, fvX6, fvy6 = preprowf(path,'all(VFT)',1119,11)

#TMTA
# aX1, ay1, avX1, avy1 = preproa(path,'all(TMT)v',1600,5)
#aX2, ay2, avX2, avy2 = preproa(path,'all(TMT)v',1600,6)
#aX3, ay3, avX3, avy3 = preproa(path,'all(TMT)v',1600,8)
#aX4, ay4, avX4, avy4 = preproa(path,'all(TMT)v',1600,9)
#aX5, ay5, avX5, avy5 = preproa(path,'all(TMT)v',1600,11)
#aX6, ay6, avX6, avy6 = preproa(path,'all(TMT)v',1600,11)

#TMTB 
#bX1, by1, bvX1, bvy1 = preprob(path,'all(TMT)v',3600,0)
#bX2, by2, bvX2, bvy2 = preprob(path,'all(TMT)v',3600,1)
#bX3, by3, bvX3, bvy3 = preprob(path,'all(TMT)v',3600,2)
#bX4, by4, bvX4, bvy4 = preprob(path,'all(TMT)v',3600,3)
#bX5, by5, bvX5, bvy5 = preprob(path,'all(TMT)v',3600,4)
#bX6, by6, bvX6, bvy6 = preprob(path,'all(TMT)v',3600,11)

#TMTA(ori)
# aoX1, aoy1, avoX1, avoy1 = preprooria(path,'all(TMT)',1570,5)
# aoX2, aoy2, avoX2, avoy2 = preprooria(path,'all(TMT)',1570,6)
# aoX3, aoy3, avoX3, avoy3 = preprooria(path,'all(TMT)',1570,8)
# aoX4, aoy4, avoX4, avoy4 = preprooria(path,'all(TMT)',1570,9)
# aoX5, aoy5, avoX5, avoy5 = preprooria(path,'all(TMT)',1570,11)
#aoX6, ayo6, avoX6, avoy6 = preprooria(path,'all(TMT)',1570,11)

#TMTB(ori)
# boX1, boy1, bvoX1, bvoy1 = preproorib(path,'all(TMT)',3600,5)
# boX2, boy2, bvoX2, bvoy2 = preproorib(path,'all(TMT)',3600,6)
# boX3, boy3, bvoX3, bvoy3 = preproorib(path,'all(TMT)',3600,8)
# boX4, boy4, bvoX4, bvoy4 = preproorib(path,'all(TMT)',3600,9)
# boX5, boy5, bvoX5, bvoy5 = preproorib(path,'all(TMT)',3600,11)
#boX6, boy6, bvoX6, bvoy6 = preproorib(path,'all(TMT)',3600,11)
    
#TMTA+B(ori)
#aboX1, aboy1, abvoX1, abvoy1 = preprooriab(path,'all(TMT)',5)
#aboX2, aboy2, abvoX2, abvoy2 = preprooriab(path,'all(TMT)',6)
#aboX3, aboy3, abvoX3, abvoy3 = preprooriab(path,'all(TMT)',8)
#aboX4, aboy4, abvoX4, abvoy4 = preprooriab(path,'all(TMT)',9)
#aboX5, aboy5, abvoX5, abvoy5 = preprooriab(path,'all(TMT)',11)
#aboX6, abyo6, abvoX6, abvoy6 = preprooriab(path,'all(TMT)',11)

#WCST+grade mack
#wpX1, wpy1, WpvX1, wpvy1 = preprowpg(path,'all(WCST)',1119,12)
#wpX2, wpy2, WpvX2, wpvy2 = preprowpg(path,'all(WCST)',1119,14)
#wpX3, wpy3, WpvX3, wpvy3 = preprowpg(path,'all(WCST)',1119,15)
#wpX4, wpy4, WpvX4, wpvy4 = preprowpg(path,'all(WCST)',1119,9)
#wpX5, wpy5, WpvX5, wpvy5 = preprowpg(path,'all(WCST)',1119,11)
#wX6, wy6, WvX6, wvy6 = preprowf(path,'all(WCST)',1119,11)
#--------------------make train data End-------------------- 

#--------------------train、make test data & predict--------------------
for i  in range (3,11): 
    print('max_depth='+str(i))
#   WCST
#    W1p, wty1 = pred(path, wX1, wy1,'all(WCST)',5)
#    W2p, wty2 = pred(path, wX2, wy2,'all(WCST)',6)
#    W3p, wty3 = pred(path, wX3, wy3,'all(WCST)',8)
#    W4p, wty4 = pred(path, wX4, wy4,'all(WCST)',9)
#    W5p, wty5 = pred(path, wX5, wy5,'all(WCST)',11)
#    W6p, wty6 = pred(path, wX6, wy6,'all(WCST)',11)

#   WCST(50-1)
#    W1p, wty1 = pred(path, wX1, wy1,'WCST(50-1)',15)
#    W2p, wty2 = pred(path, wX2, wy2,'WCST(50-1)',14)
#    W3p, wty3 = pred(path, wX3, wy3,'WCST(50-1)',15)
#    W4p, wty4 = pred(path, wX4, wy4,'WCST(50-1)',9)
#    W5p, wty5 = pred(path, wX5, wy5,'WCST(50-1)',11)
#    W6p, wty6 = pred(path, wX6, wy6,'all(WCST)',11)
    
#   VFT
    F1p, fty1 = pred(path, fX1, fy1,'all(VFT)',5)
    F2p, fty2 = pred(path, fX2, fy2,'all(VFT)',6)
    F3p, fty3 = pred(path, fX3, fy3,'all(VFT)',8)
    F4p, fty4 = pred(path, fX4, fy4,'all(VFT)',9)
    F5p, fty5 = pred(path, fX5, fy5,'all(VFT)',11)
#    F6p, fty6 = pred(path, fX6, fy6,'all(VFT)',11)
    
#   VFT(50-1)
#    F1p, fty1 = pred(path, fX1, fy1,'VFT(50-1)',15)
#    F2p, fty2 = pred(path, fX2, fy2,'VFT(50-1)',14)
#    F3p, fty3 = pred(path, fX3, fy3,'VFT(50-1)',15)
#    F4p, fty4 = pred(path, fX4, fy4,'VFT(50-1)',9)
#    F5p, fty5 = pred(path, fX5, fy5,'VFT(50-1)',11)
#    F6p, fty6 = pred(path, fX6, fy6,'all(VFT)',11)
    
#   TMTA
#    A1p, aty1 = preda(path, aX1, ay1,'all(TMT)v',5)
#    A2p, aty2 = preda(path, aX2, ay2,'all(TMT)v',6)
#    A3p, aty3 = preda(path, aX3, ay3,'all(TMT)v',8)
#    A4p, aty4 = preda(path, aX4, ay4,'all(TMT)v',9)
#    A5p, aty5 = preda(path, aX5, ay5,'all(TMT)v',11)
#    A6p, aty6 = preda(path, aX6, ay6,'all(TMT)v',11) 
    
#   TMTB
#    B1p, bty1 = predb(path, bX1, by1,'all(TMT)v',0)
#    B2p, bty2 = predb(path, bX2, by2,'all(TMT)v',1)
#    B3p, bty3 = predb(path, bX3, by3,'all(TMT)v',2)
#    B4p, bty4 = predb(path, bX4, by4,'all(TMT)v',3)
#    B5p, bty5 = predb(path, bX5, by5,'all(TMT)v',4)
#    B6p, bty6 = predb(path, bX6, by6,'all(TMT)v',11) 
    
#    own(WCST)
#    W1p, wty1 = pred(path, wX1, wy1,'own(WCST)',1)
#    W2p, wty2 = pred(path, wX2, wy2,'own(WCST)',1)
#    W3p, wty3 = pred(path, wX3, wy3,'own(WCST)',1)
#    W4p, wty4 = pred(path, wX4, wy4,'own(WCST)',1)
#    W5p, wty5 = pred(path, wX5, wy5,'own(WCST)',1)
#    W6p, wty6 = pred(path, wX6, wy6,'own(WCST)',1) 
    
#    new(WCST)
#    W1p, wty1 = prednew(path, wX1, wy1,'new(WCST)',0)
#    W2p, wty2 = prednew(path, wX2, wy2,'own(WCST)',1)
#    W3p, wty3 = prednew(path, wX3, wy3,'own(WCST)',1)
#    W4p, wty4 = prednew(path, wX4, wy4,'own(WCST)',1)
#    W5p, wty5 = prednew(path, wX5, wy5,'own(WCST)',1)
#    W6p, wty6 = prednew(path, wX6, wy6,'own(WCST)',1) 
    
#   TMTA(ori)
    # Ao1p, aoty1 = predoria(path, aoX1, aoy1,'all(TMT)',5)
    # Ao2p, aoty2 = predoria(path, aoX2, aoy2,'all(TMT)',6)
    # Ao3p, aoty3 = predoria(path, aoX3, aoy3,'all(TMT)',8)
    # Ao4p, aoty4 = predoria(path, aoX4, aoy4,'all(TMT)',9)
    # Ao5p, aoty5 = predoria(path, aoX5, aoy5,'all(TMT)',11)
#    Ao6p, aoty6 = predoria(path, aoX6, aoy6,'all(TMT)',11)
    
#   TMTA(ori)(50-1)
#    Ao1p, aoty1 = predoria(path, aoX1, aoy1,'TMT(50-1)',12)
#    Ao2p, aoty2 = predoria(path, aoX2, aoy2,'TMT(50-1)',14)
#    Ao3p, aoty3 = predoria(path, aoX3, aoy3,'TMT(50-1)',15)
#    Ao4p, aoty4 = predoria(path, aoX4, aoy4,'TMT(50-1)',9)
#    Ao5p, aoty5 = predoria(path, aoX5, aoy5,'TMT(50-1)',11)
    
#   TMTB(ori)
    # Bo1p, boty1 = predorib(path, boX1, boy1,'all(TMT)',5)
    # Bo2p, boty2 = predorib(path, boX2, boy2,'all(TMT)',6)
    # Bo3p, boty3 = predorib(path, boX3, boy3,'all(TMT)',8)
    # Bo4p, boty4 = predorib(path, boX4, boy4,'all(TMT)',9)
    # Bo5p, boty5 = predorib(path, boX5, boy5,'all(TMT)',11)
#    Bo6p, boty6 = predorib(path, boX6, boy6,'all(TMT)',11)

#   TMTA+B(ori)
#    ABo1p, aboty1 = predoriab(path, aboX1, aboy1,'all(TMT)',5)
#    ABo2p, aboty2 = predoriab(path, aboX2, aboy2,'all(TMT)',6)
#    ABo3p, aboty3 = predoriab(path, aboX3, aboy3,'all(TMT)',8)
#    ABo4p, aboty4 = predoriab(path, aboX4, aboy4,'all(TMT)',9)
#    ABo5p, aboty5 = predoriab(path, aboX5, aboy5,'all(TMT)',11)
#    ABo6p, aboty6 = predoriab(path, aboX6, aboy6,'all(TMT)',11)
    
#   WCST+grade
#    Wp1p, wpty1 = predwpg(path, wpX1, wpy1,'all(WCST)',12)
#    Wp2p, wpty2 = predwpg(path, wpX2, wpy2,'all(WCST)',14)
#    Wp3p, wpty3 = predwpg(path, wpX3, wpy3,'all(WCST)',15)
#    Wp4p, wpty4 = predwpg(path, wpX4, wpy4,'all(WCST)',9)
#    Wp5p, wpty5 = predwpg(path, wpX5, wpy5,'all(WCST)',11)
#    W6p, wty6 = pred(path, wX6, wy6,'all(WCST)',11)
#--------------------train、make test data & predict End--------------------
    
#--------------------ensemble--------------------
    plal = (
            # Ao1p + Ao2p + Ao3p + Ao4p + Ao5p 
            +
            # Bo1p + Bo2p + Bo3p + Bo4p + Bo5p 
            +
#            ABo1p + ABo2p + ABo3p + ABo4p + ABo5p
#            + 
#            W1p + W2p + W3p + W4p + W5p 
#            +  
#            Wp1p + Wp2p + Wp3p + Wp4p + Wp5p
#            +
            F1p + F2p + F3p 
            # + F4p + F5p
#            + 
#            W1p 
#+ F1p + Ao1p + Bo1p
            )
#            A1p + A2p + A3p + A4p + A5p + B1p + B2p + B3p + B4p + B5p )
    l = plal.shape
    for i in range (0,l[0]):
        if plal[i] >= 3:
            plal[i] = 1
        else:
            plal[i] = 0
#--------------------ensemble End--------------------

#--------------------print model performance--------------------
#   ACC & AUC
#    auct = roc_auc_score(wty1, plal)
#    print("Performance sur le test auc : ", auct)
    acct = accuracy_score(fty1, plal)
    print("Performance sur le test acc : ", acct)
    kk = fty1 - plal
    print(np.where(kk != 0))
#--------------------print model performance End--------------------
#--------------------------------------------------main code End--------------------------------------------------

