import numpy as np
import random
import pandas as pd
import dnn
import gbm
import rf

def ml_cv(ytrain, xtrain,method, hyper_grid,t):
    if method=='DNN':
        ml=dnn
    if method=='GBM':
        ml=gbm_out   
    if method=='RF':
        ml=rf_out
    stepsize= np.ceil((1/3)*len(ytrain))
    nobs = min(3*stepsize,len(ytrain))
    random.seed(1)
    
    idx = [i for i in range(int(nobs))]
    random.shuffle(idx)
    sample1 = idx[0:int(stepsize)]
    sample2 = idx[int(stepsize):int(2*stepsize)]
    sample3 = idx[int(2*stepsize):]
    loss = []
    epoch_opt=[]
    for i in range(1,4):
        if i==1:
            tesample=sample1
            trsample= np.append(sample2,sample2)
        if i==2:
            tesample=sample3
            trsample= np.append(sample1,sample2)
        if i==3:
            tesample=sample2
            trsample= np.append(sample1,sample3)

            
        if method=='DNN':
            temp=dnn(ytrain[trsample],xtrain[trsample,:], ytrain[tesample],xtrain[tesample,:],hyper_grid.iloc[t,:])
            loss.append(temp[1])
            epoch_opt.append(temp.epoch_opt)
            out = np.append(hyper_grid.iloc[t,:],np.mean(loss))
            out[:,4]= np.mean(epoch_opt)
            
        if method=='GBM':
            #print(hyper_grid.iloc[t,:])
            temp=gbm_out(ytrain[trsample],xtrain[trsample,:], ytrain[tesample],xtrain[tesample,:],hyper_grid.iloc[t,:])  
            loss.append(temp[1])
            print(hyper_grid.iloc[t,:].shape)
            out= np.append(hyper_grid.iloc[t,:],np.mean(loss))
                           
        if method=='RF':
            temp=rf_out(ytrain[trsample],xtrain[trsample,:], ytrain[tesample],xtrain[tesample,:],hyper_grid.iloc[t,:])
            loss.append(temp[1])
            out=np.append(hyper_grid.iloc[t,:],np.mean(loss))
                          
    return out


    