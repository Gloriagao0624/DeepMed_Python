import numpy as np
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization


def dnn(y, x, ytest, xtest, hyper):
    train_ybin=1*(len(np.unique(y))==2 & min(y)==0 & max(y)==1)
    hyper = hyper.values
    
    hyper = hyper.T.reshape(-1)
    l1=hyper[0]
    layers=hyper[1]
    units=hyper[2]
    epochs=hyper[3]
    batch_size=hyper[4]

# define the keras model
    model = Sequential()
    i=0
    while i<layers:
        if i ==0:
            n_input = x.shape[1]
            lambda_l1=l1
            units_i= units
        else:
            n_input = units
            lambda_l1=0
            units_i=units 

        model.add(Dense(units_i, input_shape=n_input, activation='relu',kernel_regularizer = regularizer_l1(lambda_l1)))
        model.add(BatchNormalization())
        i+=1

    if train_ybin==1:
        model.add(Dense(units=1, input_shape=n_input, activation='sigmoid',kernel_initializer = initializer_glorot_uniform(1)))
        model.compile(optimizer = tf.keras.optimizers.Adam(),loss = tf.keras.losses.BinaryCrossentropy(),metrics = ['binary_crossentropy'])
    else:
        model.add(Dense(units=1, input_shape=n_input,kernel_initializer = initializer_glorot_uniform(1)))
        model.compile(optimizer = tf.keras.optimizers.Adam(),loss = tf.keras.losses.MeanSquaredError(),metrics = ['mean_squared_error'])
    
    NNfit = model.fit(x, y, epochs = epochs, batch_size=batch_size, verbose = 0,validation_data=(xtest, ytest))

    # make probability predictions with the model
    ypred = model.predict(xtest)
    val_loss_all = NNfit.hostory["val_loss"]
    loss = min(val_loss_all)
    epoch_opt = np.argmin(val_loss_all)
    
    return loss,epoch_opt,ypred