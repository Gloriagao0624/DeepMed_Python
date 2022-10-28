import numpy as np
import random
import pandas as pd
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import Model
 


def make_dnn(train_ybin,input_dim,l1,layers,units):
    inputs = Input(shape=(input_dim,), name='input')
    phi = Dense(units=units, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=1),name='phi_1')(inputs)
    for i in range(layers):
        phi = Dense(units=units, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=1),name='phi_'+str(i+2))(phi)
        phi = BatchNormalization()(phi)
    
    
    if train_ybin==1:
        phi = Dense(units=1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=1))(phi)
        model = Model(inputs=inputs, outputs=phi)
        #model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['binary_crossentropy'])
    else:
        phi = Dense(units=1,kernel_initializer=initializers.glorot_uniform(seed=1))(phi)
        model = Model(inputs=inputs, outputs=phi)
        #model.compile(optimizer='adam',loss = "mean_squared_error",metrics = ['mean_squared_error'])
        
    return model

def dnn(y, x, ytest, xtest, hyper):
    train_ybin=1*((len(np.unique(y))==2) & (min(y)==0) & (max(y)==1))
    l1=hyper[0]
    layers=int(hyper[1])
    units=int(hyper[2])
    epochs=math.ceil(hyper[3])
    batch_size=int(hyper[4])
    input_dim = x.shape[1]
    NN = make_dnn(train_ybin,input_dim,l1,layers,units)
    if train_ybin==1:
        NN.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['binary_crossentropy'])
    else:
        NN.compile(optimizer='adam',loss = "mean_squared_error",metrics = ['mean_squared_error'])
    NNfit = NN.fit(x, y, epochs =epochs, batch_size=batch_size, verbose = True,validation_data=(xtest, ytest))   
        
    ypred = NN(xtest).numpy().flatten()
    val_loss_all = NNfit.history["val_loss"]
    loss = min(val_loss_all)
    epoch_opt = np.argmin(val_loss_all)
    return loss,ypred,epoch_opt   
    
    
    