import numpy as np
import random
import pandas as pd

def DeepMed_bin_cv(y,d,m,x,method,hyper_grid,epochs,batch_size):
      
    if method=="DNN":
        hyper_grid= pd.concat([hyper_grid, epochs,batch_size], axis=1)
        
    n_hyper= hyper_grid.shape[1]
    stepsize= np.ceil((1/3)*len(d))
    
    random.seed(1)
    
    idx = [i for i in range(len(d))]
    random.shuffle(idx)
     # crossfitting procedure that splits sample in training an testing data
    for k in range(1,4):
        tesample=idx[int((k-1)*stepsize+1):int(min((k)*stepsize,len(d)))]
        #dtr=d[-tesample]
        dtr = np.delete(d,tesample)
        dte = d[tesample]
        
        ytr=np.delete(y,tesample)
        yte=y[tesample]
        
        ytr1=ytr[dtr==1]
        ytr0=ytr[dtr==0]
        
        mtr=np.delete(m,tesample)
        mte=m[tesample]
        
        mtr1=mtr[dtr==1]
        mtr0=mtr[dtr==0]
        

        if (x.shape[1] is None or x.shape[1]==1):
            xtr= np.delete(x,tesample)
            xte=x[tesample]
            xtr1=xtr[dtr==1]
            xtr0=xtr[dtr==0]
            xtr11=xtr[(dtr==1) & (mtr==1)]
            xtr10=xtr[(dtr==1) & (mtr==0)]
            xtr01=xtr[(dtr==0) & (mtr==1)]
            xtr00=xtr[(dtr==0) & (mtr==0)]

        if ((x.shape[1] == 0) and x.shape[1]>1):
            xte = x[tesample,:]
            xtr = np.setdiff1d(x, xtr).view(x.dtype).reshape(-1, x.shape[1]).shape
            xtr1=xtr[dtr==1,:]
            xtr0=xtr[dtr==0,:]
            xtr11=xtr[(dtr==1) & (mtr==1),:]
            xtr10=xtr[(dtr==1) & (mtr==0),:]
            xtr01=xtr[(dtr==0) & (mtr==1),:]
            xtr00=xtr[(dtr==0) & (mtr==0),:]

        ytr11=ytr[(dtr==1) & (mtr==1)]
        ytr10=ytr[(dtr==1) & (mtr==0)]
        ytr01=ytr[(dtr==0) & (mtr==1)]
        ytr00=ytr[(dtr==0) & (mtr==0)]
        # tr stands for first training data, te for test data, "1" and "0" for subsamples with treated and nontreated


        # predict Pr(M=1|D=1,X) in test data
        # predict Pr(M=1|D=0,X) in test data
        # predict Pr(D=1|X) in test data
        # predict E(Y| D=1, M=1, X) in test data
        # predict E(Y| D=0, M=1, X) in test data
        # predict E(Y| D=1, M=0, X) in test data
        # predict E(Y| D=0, M=0, X) in test data
        # predict E(Y|D=1, X) in test data
        # predict E(Y|D=0, X) in test data

        # Get all worker processes
        cores = multiprocess.cpu_count()
        # Start all worker processes
        pool = multiprocess.Pool(processes=cores)
        y1 = [mtr1,mtr0,dtr,ytr11,ytr01,ytr10,ytr00,ytr1,ytr0]
        x1 = [xtr1,xtr0,xtr,xtr11,xtr01,xtr10,xtr00,xtr1,xtr0]

        tasks = [(y,x) for y in y1 for x in x1]
        out = pool.starmap(ml_cv,tasks)
        for i in range(1,10):
            outi=out[:,1:(n_hyper+1)]
            out=out[:,n_hyper+1:]
            loc = np.argmin(outi[:,(n_hyper+1)])
            hyper_k = pd.concat([hyper_k,outi[loc,:]], axis=1)
            
                
        for i in range(9):
            outi=out[:,0:n_hyper]
            out=out[:,n_hyper+1:out.shape[1]]
            loc = np.argmin(outi[:, n_hyper])
            hyper_k = hyper_k.append(outi[loc,:])

        hyper_k.columns =  np.arange(1,10) 
        hyper_k.loc[n_hyper,:]=round(hyper_k.loc[n_hyper,:],3)
        
        if method=="DNN":
            hyper_k.loc[3,:]=round(hyper_k.loc[3,:])

        if k==1:                      
            hyper=hyper_k
        else:
            hyper = pd.concat([hyper,hyper_k], axis=1) 
                                   
    return hyper