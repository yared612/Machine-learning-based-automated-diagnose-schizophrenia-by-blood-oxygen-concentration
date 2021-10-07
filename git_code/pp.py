#補植
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

test_file   = 'D:\\HCSZ\\all(WCST)\\ch1_16\\val'
p_file = 'D:\\HCSZ\\all(only WCST grade)\\ch1_16\\val'
class_ = glob.glob(test_file + '\\*')
class_name = []
for name in class_:
    class_name.append(name.split('\\')[-1])

for g  in range(0,2):
    filename = glob.glob(test_file + '\\' + class_name[g] +'\\*.csv')
    im_name = []
    for i in filename:
        n = i.split('\\')[-1].split('.')[0]
        im_name.append(n)
    for file in im_name:
            data = pd.read_csv(test_file + '\\' + class_name[g] + '\\' + file + '.csv')
            data1 = pd.read_csv(p_file + '\\' + class_name[g] + '\\' + file + '.csv')
            for k in range (data.shape[0],1160):
                data.loc[k] = np.array(data1)[0,1]/122
            data.to_csv('D:\\HCSZ\\all(plus WCST grade)\\ch1_16\\val' + '\\' + class_name[g] + '\\' + file + '.csv',encoding='utf8',index = False)