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

train_file = 'D:\\HCSZ\\all(WCST)\\ch1_16\\train'
val_file   = 'D:\\HCSZ\\all(WCST)\\ch1_16\\val'
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
#        pg = pd.read_csv(data_file + 'pn'+ '/' + file + '.csv')
#        data.loc[data.shape[0]+1] = pg.iat[0,1]
        data = np.array(data)[:,1]
        data = np.reshape(data,data.shape+(1,))
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
        data_split_out.append(c)
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
            l = l/.3
#            l = (l - l.min())/(l.max() - l.min())
            out.append(l)
    if mode == 'std':
        for l in list_:
            l = (l - l.mean())/l.std()
            out.append(l)
    return out

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
good_y = np.zeros([len(good_X)])
bad_y  = np.ones([len(bad_X)])
train_y_z = np.concatenate([good_y,bad_y])
train_y_o = np.stack([1-train_y_z,train_y_z],axis=-1)
val_good_y = np.zeros([len(val_good_X)])
val_bad_y  = np.ones([len(val_bad_X)])
val_y_z = np.concatenate([val_good_y,val_bad_y])
val_y_o = np.stack([1-val_y_z,val_y_z],axis=-1)
good_X.extend(bad_X), val_good_X.extend(val_bad_X)
X, y, val_X, val_y = nor(good_X,'nor') , train_y_o , nor(val_good_X,'nor') , val_y_o

model = class_model()
#model = generate_lstmfcn()
model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
#model.load_weights('D:\\HCSZ\\saved_models\\lstmCNN(all(WCST)).h5')
epochs = 600
saved_dir = './saved_models'
model_name = 'lstmCNN(all(WCST))ch2.h5'
model_path = '/'.join((saved_dir, model_name))
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1,save_best_only = True)
EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min', baseline=None)
csv_logger = CSVLogger('lstmCNN(all(WCST))ch2.log')

history = model.fit(np.array(X),np.array(y),batch_size = 4, epochs=epochs, 
                    validation_data = [np.array(val_X), np.array(val_y)],
                    shuffle = True ,
                    callbacks=[checkpoint, csv_logger])


epochs=range(len(history.history['acc']))
plt.figure()
plt.plot(epochs,history.history['acc'],'b',label='Training accuracy')
plt.plot(epochs,history.history['val_acc'],'r',label='Validation accuracy')
plt.title('Traing and Validation acc_change')
plt.legend()
plt.savefig('D:/HCSZ/figure/acc_lstmCNN(all(WCST))ch2.jpg')
plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
plt.title('Traing and Validation loss_change')
plt.legend()
plt.savefig('D:/HCSZ/figure/loss_lstmCNN(all(WCST))ch2.jpg')


