from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
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

train_file = '/home/yared/文件/HCSZ/all(WCST)/ch1_16/train'
val_file   = '/home/yared/文件/HCSZ/all(WCST)/ch1_16/val'

#--------------------------------------------------catch category--------------------------------------------------
class_ = glob.glob(train_file + '/*')
class_name = []
for name in class_:
    class_name.append(name.split('/')[-1])
#--------------------------------------------------catch category End--------------------------------------------------

#--------------------------------------------------function--------------------------------------------------
#--------------------內插--------------------
def data_aug1(data_file,crop_number = 10):
    filename = glob.glob(data_file + '*.csv')
    im_name = []
    for i in filename:
        n = i.split('/')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    for file in im_name:
        data = pd.read_csv(data_file + file + '.csv')
#        data1 = pd.read_csv(data_file + file + '.csv')
        pg = pd.read_csv('/home/yared/文件/HCSZ/all(only WCST grade)/ch1_16/' + data_file.split('/')[-3] + '/' + 
                         data_file.split('/')[-2] + '/' + file + '.csv')
        data = np.array(data)[:,0]
        pg  = np.array(pg)[:,1]
#        data1 = np.array(data1)[:,0]
#        data = np.concatenate((data, data1), axis=0)
#        data = np.concatenate((data, pg), axis=0)
        data = np.reshape(data,data.shape+(1,))
        pg = np.reshape(pg,pg.shape+(1,))
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
            x1 = np.linspace(0,len(c0)-1,1119)
            y0 = np.interp(x1,x0,c0)
            c.append(y0)            
        c = np.stack(c,axis=-1)
        c = np.concatenate((c,pg), axis = 0)
        data_split_out.append(c.T)
    return data_split_out
#--------------------內插 End--------------------
    
#--------------------讀圖頻率(aug用)--------------------
def frequency(data_file,num,crop_number):
    final = []
    for i in range(1,num):
        x = data_aug1(data_file,crop_number)
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
            pg = pd.read_csv('/home/yared/文件/HCSZ/all(only WCST grade)/ch1_16/' + data_file.split('/')[-3] + '/' + 
                         data_file.split('/')[-2] + '/' + file + '.csv')
            data = np.array(data)[:,0]
#            data1 = np.array(data1)[:,15]
            pg  = np.array(pg)[:,1]
#            data = np.concatenate((data, data1), axis=0)
#            data = np.concatenate((data, pg), axis=0)
            data = np.reshape(data,data.shape+(1,))
            pg = np.reshape(pg,pg.shape+(1,))
            data = np.concatenate((data, pg), axis=0)
            aa.append(data.T)  
        for k in range(0,len(aa)):
            data_split_out.append(aa[k])
        return data_split_out
#--------------------read test data End--------------------
#--------------------------------------------------function End--------------------------------------------------
        
#--------------------------------------------------main code--------------------------------------------------
#--------------------make train and val data--------------------
good_x1 = frequency(train_file + '/' + class_name[0] + '/', 2, 20)
good_x2 = frequency(train_file + '/' + class_name[0] + '/', 4, 25)
good_x3 = frequency(train_file + '/' + class_name[0] + '/', 6, 30)
good_X  = good_x1 + good_x2 + good_x3
bad_x1 = frequency(train_file + '/' + class_name[1] + '/', 2, 20)
bad_x2 = frequency(train_file + '/' + class_name[1] + '/', 4, 25)
bad_x3 = frequency(train_file + '/' + class_name[1] + '/', 6, 30)
bad_X = bad_x1 + bad_x2 + bad_x3
val_good_x1 = frequency(val_file + '/' + class_name[0] + '/', 2, 20)
val_good_x2 = frequency(val_file + '/' + class_name[0] + '/', 4, 30)
val_good_x3 = frequency(val_file + '/' + class_name[0] + '/', 6, 30)
val_good_X = val_good_x1 + val_good_x2 + val_good_x3
val_bad_x1 = frequency(val_file + '/' + class_name[1] + '/', 2, 30)
val_bad_x2 = frequency(val_file + '/' + class_name[1] + '/', 4, 30)
val_bad_x3 = frequency(val_file + '/' + class_name[1] + '/', 6, 30)
val_bad_X = val_bad_x1 + val_bad_x2 + val_bad_x3
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
#--------------------make train and val data End--------------------
    
#--------------------train、make test data & predict--------------------
#"for" is to set up the max_depth
for i  in range (3,11): 
#train
    print('max_depth='+str(i))
    df1 = df1.dropna(axis=0,how='any')
    my_model = XGBClassifier(max_depth=i, colsample_bytree = 0.7, eta = 0.2, min_child_weight = 3 )
    # Add silent=True to avoid printing out updates with each cycle
    my_model.fit(df, y, verbose=False)
    my_model.fit(df, y, verbose=False)
    train_y_pred = my_model.predict(df)
#print auc & acc for train and val
    auc = roc_auc_score(y, train_y_pred)
    print("Performance sur le train auc : ", auc)
    acc = accuracy_score(y, train_y_pred)
    print("Performance sur le train acc : ", acc)   
    val_y_pred1 = my_model.predict(df1)
    aucv = roc_auc_score(val_y, val_y_pred1)
    print("Performance sur le val auc : ", aucv)
    accv = accuracy_score(val_y, val_y_pred1)
    print("Performance sur le val acc : ", accv)
#show feature important    
    #plot_importance(my_model)
    #pyplot.show()
#read test    
#     test_file   = '/home/yared/文件/HCSZ/all(WCST)/ch1_16/test'        
#     gx = read_data(test_file + '/' + class_name[0] + '/')
#     bx = read_data(test_file + '/' + class_name[1] + '/')
#     good_yt = np.zeros([len(gx)])
#     bad_yt  = np.ones([len(bx)])
#     test_y_z = np.concatenate([good_yt,bad_yt])
#     gx.extend(bx)
#     test_X, test_y = nor(gx,'nor') , test_y_z
    
#     tt = test_X[0]
#     df2 = pd.DataFrame(tt)
#     for k in range (1,len(test_X)):
#         ttt = test_X[k]
#         df2 = pd.concat([df2, pd.DataFrame(ttt)], axis=0)
#     df2 = df2.dropna(axis=0,how='any')
# #predict
#     val_y_pred2 = my_model.predict(df2)
# #print auc & acc for test
#     auct = roc_auc_score(test_y, val_y_pred2)
#     print("Performance sur le test auc : ", auct)
#     acct = accuracy_score(test_y, val_y_pred2)
#     print("Performance sur le test acc : ", acct)
#     kk = test_y - val_y_pred2
#     print(np.where(kk != 0))
#--------------------train、make test data & predict End--------------------
#--------------------------------------------------main code End-------------------------------------------------- 
    
#--------------------------------------------------plot signal img--------------------------------------------------
    # plt.plot(gg)
    # plt.show()
#--------------------------------------------------plot signal img End--------------------------------------------------