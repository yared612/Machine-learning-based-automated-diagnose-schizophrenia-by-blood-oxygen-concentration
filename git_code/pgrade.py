from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from matplotlib import pyplot
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
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

train_file = 'D:\\HCSZ\\all(WCST grade)\\ch1_16\\train'
val_file   = 'D:\\HCSZ\\all(WCST grade)\\ch1_16\\val'
class_ = glob.glob(train_file + '\\*')
class_name = []
for name in class_:
    class_name.append(name.split('\\')[-1])

def data_aug1(data_file,crop_number = 10):
    filename = glob.glob(data_file + '*.csv')
    im_name = []
    for i in filename:
        n = i.split('\\')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    for file in im_name:
        data = pd.read_csv(data_file + file + '.csv')
#        data1 = pd.read_csv(data_file + file + '.csv')
#        pg = pd.read_csv(data_file + 'pn'+ '\\' + file + '.csv')
        data = np.array(data)[:,1]
#        pg  = np.array(pg)[]
#        data1 = np.array(data1)[:,0]
#        data = np.concatenate((data, data1), axis=0)
#        data = np.concatenate((data, pg), axis=0)
        data = np.reshape(data,data.shape+(1,))
#        
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
            x1 = np.linspace(0,len(c0)-1,1)
            y0 = np.interp(x1,x0,c0)
            c.append(y0)            
        c = np.stack(c,axis=-1)
        data_split_out.append(c.T)
    return data_split_out
def frequency(data_file,num,crop_number):
    final = []
    for i in range(1,num):
        x = data_aug1(data_file,crop_number)
        final+=x     
    return final

def nor(list_,mode = 'nor'):
    out = []
    if mode == 'nor':
        for l in list_:
            l = l/1
#            l = (l - l.min())/(l.max() - l.min())
            out.append(l)
    if mode == 'std':
        for l in list_:
            l = (l - l.mean())/l.std()
            out.append(l)
    return out
def read_data(data_file):
    filename = glob.glob(data_file + '*.csv')
    im_name = []
    for i in filename:
        n = i.split('\\')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    aa = []
    for file in im_name:
        data = pd.read_csv(data_file + file + '.csv')
#            data1 = pd.read_csv(data_file + file + '.csv')
#            pg = pd.read_csv(data_file + 'pn'+ '\\' + file + '.csv')
        data = np.array(data)[:,1]
#            data1 = np.array(data1)[:,15]
#            pg  = np.array(pg)[]
#            data = np.concatenate((data, data1), axis=0)
#            data = np.concatenate((data, pg), axis=0)
        data = np.reshape(data,data.shape+(1,))
        aa.append(data.T)  
    for k in range(0,len(aa)):
        data_split_out.append(aa[k])
    return data_split_out

good_X = read_data(train_file + '\\' + class_name[0] + '\\')
bad_X = read_data(train_file + '\\' + class_name[1] + '\\')
val_good_X = read_data(val_file + '\\' + class_name[0] + '\\')
val_bad_X = read_data(val_file + '\\' + class_name[1] + '\\')
good_y = np.zeros(len(good_X))
bad_y  = np.ones(len(bad_X))
train_y_z = np.concatenate([good_y,bad_y])
val_good_y = np.zeros([len(val_good_X)])
val_bad_y  = np.ones([len(val_bad_X)])
val_y_z = np.concatenate([val_good_y,val_bad_y])
good_X.extend(bad_X), val_good_X.extend(val_bad_X)
X, y, val_X, val_y = nor(good_X,'nor') , train_y_z , nor(val_good_X,'nor') , val_y_z

ss = X[0]
df = pd.DataFrame(ss)
for i in range (1,len(X)):
    sss = X[i]
    df = pd.concat([df, pd.DataFrame(sss)], axis=0)
#df = df.dropna(axis=0,how='any')
vv = val_X[0]
df1 = pd.DataFrame(vv)
for j in range (1,len(val_X)):
    vvv = val_X[j]
    df1 = pd.concat([df1, pd.DataFrame(vvv)], axis=0)
for i  in range (1,11): 
    print('max_depth='+str(i))
    df1 = df1.dropna(axis=0,how='any')
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    # Add silent=True to avoid printing out updates with each cycle
    my_model.fit(df, y, verbose=False)
    my_model.fit(df, y, verbose=False)
    train_y_pred = my_model.predict(df)
    auc = roc_auc_score(y, train_y_pred)
    print("Performance sur le train auc : ", auc)
    acc = accuracy_score(y, train_y_pred)
    print("Performance sur le train acc : ", acc)
    
    val_y_pred1 = my_model.predict(df1)
    aucv = roc_auc_score(val_y, val_y_pred1)
    print("Performance sur le val auc : ", aucv)
    accv = accuracy_score(val_y, val_y_pred1)
    print("Performance sur le val acc : ", accv)
    
    #plot_importance(my_model)
    #pyplot.show()
    
    test_file   = 'D:\\HCSZ\\all(WCST grade)\\ch1_16\\test'            
    gx = read_data(test_file + '\\' + class_name[0] + '\\')
    bx = read_data(test_file + '\\' + class_name[1] + '\\')
    good_yt = np.zeros([len(gx)])
    bad_yt  = np.ones([len(bx)])
    test_y_z = np.concatenate([good_yt,bad_yt])
    gx.extend(bx)
    test_X, test_y = nor(gx,'nor') , test_y_z
    
    tt = test_X[0]
    df2 = pd.DataFrame(tt)
    for k in range (1,len(test_X)):
        ttt = test_X[k]
        df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
    df2 = df2.dropna(axis=0,how='any')   
    val_y_pred2 = my_model.predict(df2)
    auct = roc_auc_score(test_y, val_y_pred2)
    print("Performance sur le test auc : ", auct)
    acct = accuracy_score(test_y, val_y_pred2)
    print("Performance sur le test acc : ", acct)
    kk = test_y - val_y_pred2
    print(np.where(kk != 0))
    
#    plt.plot(gg)
#    plt.show()
