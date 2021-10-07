import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout, MaxPooling1D, Conv1D, MaxPool1D, GRU, Bidirectional, ConvLSTM2D, Permute, GlobalAveragePooling1D, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, TimeDistributed, Embedding
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.layers as KL
import numpy as np


def lstm(kernals = 2):
    x = KL.LSTM(kernals,activation='tanh',recurrent_activation='sigmoid',
                kernel_initializer='glorot_uniform',
                use_bias=True, unit_forget_bias=True,
                kernel_regularizer=None, recurrent_regularizer=None,
                return_sequences=True,return_state=False
                )
    return x
def gru(kernals = 2):
    x = KL.GRU(kernals,activation = "tanh",
               recurrent_activation = "hard_sigmoid",
               kernel_initializer = "glorot_uniform",
               use_bias=True, kernel_regularizer=None,
               recurrent_regularizer=None, return_sequences=True,
               return_state=False
               )
    return x
class AttentionBlock(Model):
    def __init__(self, h_dim):
        super(AttentionBlock, self).__init__(name='')
        self.dense1 = KL.Dense(h_dim)
        self.bn1 = KL.BatchNormalization()
        self.relu = KL.ELU()
        
        self.dense2 = KL.Dense(1)
        self.sm = KL.Softmax()
        self.d = KL.Dot(0)
    def call(self, input, training=False):
        x = self.dense1(input)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.dense2(x)
        x = tf.squeeze(x,-1)
        x = self.sm(x)
        return self.d([input, x])
class Attention(Model):
    def __init__(self,h_dim):
        super(Attention, self).__init__(name='')
        self.a1 = AttentionBlock(h_dim)
        self.a2 = AttentionBlock(h_dim)
    def call(self, input, training=False):
        f, b = tf.split(input, 2, -1)
        f = self.a1(f)
        b = self.a2(b)
        return tf.concat([f, b], -1)
   
# def class_model():
#     input_layer = KL.Input(shape=(150, 15), name='data')
#     x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
#     x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
#     x = Dropout(0.5)(x)
#     x = MaxPool1D(pool_size=2)(x)
#     x = lstm(64)(x)
#     x = Flatten()(x)
#     x = Dense(2400, activation='relu',kernel_initializer = 'he_normal')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(240,  activation='relu',kernel_initializer = 'he_normal')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(2, activation='softmax',kernel_initializer = 'he_normal')(x)
#     model = Model(inputs=input_layer, outputs=x)

#     return model


def class_model():
    input_layer = KL.Input(shape=(1119,1), name='data')
    l1 = lstm()(input_layer)
    l        = int(4) 
    one      = KL.Conv1D(l,3,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l1)
    one      = KL.BatchNormalization()(one)
    two      = KL.Conv1D(l,5,strides=1,padding='same',dilation_rate=2,kernel_initializer = 'he_normal')(l1)
    two      = KL.BatchNormalization()(two)
    three    = KL.Conv1D(l,7 ,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l1)
    three    = KL.BatchNormalization()(three)
    four     = KL.Conv1D(l,11,strides=1,padding='same',kernel_initializer = 'he_normal')(l1)
    four     = KL.BatchNormalization()(four)
    five     = KL.Conv1D(l,13,strides=1,padding='same',kernel_initializer = 'he_normal')(l1)
    five     = KL.BatchNormalization()(five)
    one      = KL.concatenate([one,two,three,four,five],axis = -1)
    one      = KL.BatchNormalization()(one)
    one      = KL.PReLU()(one)
    one_pool = KL.MaxPooling1D(pool_size=2, strides=None)(one)
    l2 = lstm(400)(one_pool)
    one      = KL.Conv1D(l*2,2,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l2)
    one      = KL.BatchNormalization()(one)
    two      = KL.Conv1D(l*2,3,strides=1,padding='same',dilation_rate=2,kernel_initializer = 'he_normal')(l2)
    two      = KL.BatchNormalization()(two)
    three    = KL.Conv1D(l*2,5 ,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l2)
    three    = KL.BatchNormalization()(three)
    four     = KL.Conv1D(l*2,7,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l2)
    four     = KL.BatchNormalization()(four)
    five     = KL.Conv1D(l*2,9,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l2)
    five     = KL.BatchNormalization()(five)
    two      = KL.concatenate([one,two,three,four,five],axis = -1)
    two      = KL.BatchNormalization()(two)
    two      = KL.PReLU()(two)
    two_pool = KL.MaxPooling1D(pool_size=2, strides=None)(two)
    l3 = lstm(300)(two_pool)
    one      = KL.Conv1D(l*4,2,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l3)
    one      = KL.BatchNormalization()(one)
    two      = KL.Conv1D(l*4,3,strides=1,padding='same',dilation_rate=2,kernel_initializer = 'he_normal')(l3)
    two      = KL.BatchNormalization()(two)
    three    = KL.Conv1D(l*4,4 ,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l3)
    three    = KL.BatchNormalization()(three)
    four     = KL.Conv1D(l*4,5,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l3)
    four     = KL.BatchNormalization()(four)
    five     = KL.Conv1D(l*4,7,strides=1,dilation_rate=2,padding='same',kernel_initializer = 'he_normal')(l3)
    five     = KL.BatchNormalization()(five)
    three    = KL.concatenate([one,two,three,four,five],axis = -1)
    three    = KL.BatchNormalization()(three)
    three    = KL.PReLU()(three)
    three_pool = KL.MaxPooling1D(pool_size=2, strides=None)(three)
    fl       = KL.Flatten()(three_pool)
    fl       = KL.Dropout(0.2)(fl)
    D5       = KL.Dense(512,kernel_initializer = 'he_normal')(fl)
    D5       = KL.PReLU()(D5)
    D5       = KL.Dropout(0.4)(D5)
    D0       = KL.Dense(256,kernel_initializer = 'he_normal')(D5)
    D0       = KL.PReLU()(D0)
    D0       = KL.Dropout(0.4)(D0)
    D1       = KL.Dense(64,kernel_initializer = 'he_normal')(D0)
    D1       = KL.PReLU()(D1)
    D1       = KL.Dropout(0.4)(D1)
    D2       = KL.Dense(10,kernel_initializer = 'he_normal')(D1)
    D2       = KL.PReLU()(D2)
    D2       = KL.Dropout(0.4)(D2)
    D3       = KL.Dense(2,activation='softmax',kernel_initializer = 'he_normal')(D2)
    model = Model(inputs=input_layer, outputs=D3)
    return model


# def BidGru():                                                            single bidGru(V1)
#     input_layer = KL.Input(shape=(200,16), name='data')
#     x = Bidirectional(gru(16))(input_layer)
#     x = Flatten()(x)
#     x = Dense(320, activation='relu',kernel_initializer = 'he_normal')(x)
#     x = Dropout(0.3)(x)
#     x = Dense(32, activation='relu',kernel_initializer = 'he_normal')(x)
#     x = Dropout(0.3)(x)
#     x = Dense(2, activation='softmax',kernel_initializer = 'he_normal')(x)
#     model = Model(inputs=input_layer, outputs=x)
#     return model

# def BidGru():                                                              #(V2)
#     input_layer = KL.Input(shape=(200,1), name='data')
#     x = Bidirectional(gru(16))(input_layer)
#     x = Dropout(0.3)(x)
#     x = Bidirectional(gru(8))(x)
#     x = Flatten()(x)
#     x = Dense(320, activation='relu',kernel_initializer = 'he_normal')(x)
#     x = Dropout(0.3)(x)
#     x = Dense(32, activation='relu',kernel_initializer = 'he_normal')(x)
#     x = Dropout(0.3)(x)
#     x = Dense(2, activation='softmax',kernel_initializer = 'he_normal')(x)
#     model = Model(inputs=input_layer, outputs=x)
#     return model
        
#def generate_lstmfcn():
#    input_layer = KL.Input(shape=(200,1), name='data')
#    x = LSTM(16)(input_layer)
#    x = Dropout(0.5)(x)
#    
#    y = Permute((1,2))(input_layer)
#    y = Conv1D(128, 8, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
#    y = Conv1D(256, 5, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
#    y = Conv1D(128, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
#    y = GlobalAveragePooling1D()(y)
#    x = concatenate([x,y])
    # out = Dense(2, activation='softmax')(x)
#    model = Model(inputs=input_layer, outputs=out)
#    return model
    

#def embedding_lstm():
#    input_layer = KL.Input(shape=(200,1), name='data')
#
#    model = Model(inputs=x, outputs=y)
#    return model
#model = embedding_lstm()
#model.summary(line_length=100)

#
# model = class_model()
# model.summary()  


