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


train_file = 'D:\\HCSZ\\all(TMT)\\ch1_16\\train'
val_file   = 'D:\\HCSZ\\all(TMT)\\ch1_16\\val'
class_ = glob.glob(train_file + '\\*')
class_name = []
for name in class_:
    class_name.append(name.split('\\')[-1])

def data_aug1(data_file,crop_number = 10):
    filename = glob.glob(data_file + 'A\\' + '*.csv')
    im_name = []
    for i in filename:
        n = i.split('\\')[-1].split('.')[0]
        im_name.append(n)
    data_split_out = []
    for file in im_name:
        data = pd.read_csv(data_file + 'A\\' + file + '.csv')
        data1 = pd.read_csv(data_file + 'B\\' + file + '.csv')
        data = np.array(data)[:,0]
        data1 = np.array(data1)[:,0]
#        data = np.concatenate((data, data1), axis=0)
        data = np.reshape(data,data.shape+(1,))
        data1 = np.reshape(data1,data1.shape+(1,))
#        pg = pd.read_csv(data_file + 'pn'+ '/' + file + '.csv')
#        data.loc[data.shape[0]+1] = pg.iat[0,1]
#        data = np.array(data)
#        number = int(len(data)/crop_number)
#A 
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
            l = l/.5
#            l = (l - l.min())/(l.max() - l.min())
            out.append(l)
    if mode == 'std':
        for l in list_:
            l = (l - l.mean())/l.std()
            out.append(l)
    return out
good_x1 = frequency(train_file + '\\' + class_name[0] + '\\', 2, 2)
good_x2 = frequency(train_file + '\\' + class_name[0] + '\\', 4, 4)
good_x3 = frequency(train_file + '\\' + class_name[0] + '\\', 6, 6)
good_X  = good_x1 + good_x2 + good_x3
bad_x1 = frequency(train_file + '\\' + class_name[1] + '\\', 2, 2)
bad_x2 = frequency(train_file + '\\' + class_name[1] + '\\', 4, 4)
bad_x3 = frequency(train_file + '\\' + class_name[1] + '\\', 6, 6)
bad_X = bad_x1 + bad_x2 + bad_x3
val_good_x1 = frequency(val_file + '\\' + class_name[0] + '\\', 2, 2)
val_good_x2 = frequency(val_file + '\\' + class_name[0] + '\\', 4, 4)
val_good_x3 = frequency(val_file + '\\' + class_name[0] + '\\', 6, 6)
val_good_X = val_good_x1 + val_good_x2 + val_good_x3
val_bad_x1 = frequency(val_file + '\\' + class_name[1] + '\\', 2, 2)
val_bad_x2 = frequency(val_file + '\\' + class_name[1] + '\\', 4, 4)
val_bad_x3 = frequency(val_file + '\\' + class_name[1] + '\\', 6, 6)
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
for i  in range (3,11): 
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
    
    test_file   = 'D:\\HCSZ\\all(TMT)\\ch1_16\\test'        
    def read_data(data_file):
        filename = glob.glob(data_file + 'A\\' + '*.csv')
        im_name = []
        for i in filename:
            n = i.split('\\')[-1].split('.')[0]
            im_name.append(n)
        data_split_out = []
        aa = []
        for file in im_name:
            data = pd.read_csv(data_file + 'A\\' + file + '.csv')
            data1 = pd.read_csv(data_file + 'B\\' + file + '.csv')
#            data1 = pd.read_csv('D:\\HCSZ\\all(WCST)\\ch1_16\\' + data_file.split('\\')[-3] + 
#                            '\\' + data_file.split('\\')[-2] + '\\' + file + '.csv')
            data = np.array(data)[:,0]
            data1 = np.array(data1)[:,0]
            data = np.concatenate((data, data1), axis=0)
            data = np.reshape(data,data.shape+(1,))
            aa.append(data.T)  
        for k in range(0,len(aa)):
            data_split_out.append(aa[k])
        return data_split_out
    
    gx = frequency(test_file + '\\' + class_name[0] + '\\', 2, 2)
    bx = frequency(test_file + '\\' + class_name[1] + '\\', 2, 2)
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
