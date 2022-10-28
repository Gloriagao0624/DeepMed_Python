import numpy as np
import random
import pandas as pd
import multiprocess
from  deepmed.dnn import dnn
from deepmed.gbm import gbm_out
from deepmed.rf import rf_out
from deepmed.lasso import ls_out

def DeepMed_bin(y,d,m,x,method,hyper,trim=0.05):
    if method=='DNN':
        ml=dnn
    if method=='GBM':
        ml=gbm_out   
    if method=='RF':
        ml=rf_out
    if method=='Lasso':
        ml=ls_out
    
    
    stepsize= np.ceil((1/3)*len(d))
    random.seed(1)
    
    idx = [i for i in range(len(d))]
    random.shuffle(idx)
    # crossfitting procedure that splits sample in training an testing data
    y1m0=list()
    y1m1=list()
    y0m0=list()
    y0m1=list()
    selall=list()
    for k in range(1,4):
        tesample=idx[int((k-1)*stepsize+1):int(min((k)*stepsize,len(d)))]
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

        if (x.shape[1] == 0 and x.shape[1]>1):
            xte = x[tesample,:]
            xtr = np.setdiff1d(x, xtr).view(x.dtype).reshape(-1, x.shape[1]).shape
            xtr1=xtr[dtr==1,:]
            xtr0=xtr[dtr==0,:]
            xtr11=xtr[(dtr==1) & (mtr==1),:]
            xtr10=xtr[(dtr==1) & (mtr==0),:]
            xtr01=xtr[(dtr==0) & (mtr==1),:]
            xtr00=xtr[(dtr==0) & (mtr==0),:]
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
        ytr11=ytr[(dtr==1) & (mtr==1)]
        ytr10=ytr[(dtr==1) & (mtr==0)]
        ytr01=ytr[(dtr==0) & (mtr==1)]
        ytr00=ytr[(dtr==0) & (mtr==0)]
        
        # Get all worker processes
        cores = multiprocess.cpu_count()
        # Start all worker processes
        pool = multiprocess.Pool(processes=cores)
        random.seed(1)
        a= [mr1,mtr0,dtr,ytr11,ytr01,ytr10,ytr00,ytr1,ytr0]
        b= [xtr1,xtr0,xtr,xtr11,xtr01,xtr10,xtr00,xtr1,xtr0]
        c=[mte,mte,dte,yte,yte,yte,yte,yte,yte]
        d = [xte,xte,xte,xte,xte,xte,xte,xte,xte]
        e = [hyper[:,0],hyper[:,1],hyper[:,2],hyper[:,3],hyper[:,4],hyper[:,5],hyper[:,6],hyper[:,7],hyper[:,8]]
        
        out = pool.map(ml, zip(a,b,c,d,e))
        pool.close()
        pm1te = out[:,0][1]
        pm0te = out[:,1][1]
        pdte = out[:,2][1]
        eymx11te = out[:,3][1]
        eymx01te = out[:,4][1]
        eymx10te = out[:,5][1]
        eymx00te = out[:,6][1]
        eyx1te = out[:,7][1]
        eyx0te = out[:,8][1]
        hyper=hyper[:,9:]
        
        # predict E(Y| D=0, M, X) in test data
        eymx0te=mte*eymx01te+(1-mte)*eymx00te
        # predict E(Y| D=1, M, X) in test data
        eymx1te=mte*eymx11te+(1-mte)*eymx10te
        
        # predict score functions for E(Y(1,M(0))) in the test data
        eta10=(eymx11te*pm0te+eymx10te*(1-pm0te))
        sel= 1*(((pdte*pm1te)>=trim) & ((1-pdte)>=trim)  & (pdte>=trim) &  (((1-pdte)*pm0te)>=trim))
        temp=dte*pm0te/(pdte*pm1te)*(yte-eymx1te)+(1-dte)/(1-pdte)*(eymx1te-eta10)+eta10
        y1m0= y1m0.append(temp[sel==1])
        # predict score functions for E(Y(1,M(1))) in the test data
        temp=eyx1te + dte*(yte-eyx1te)/pdte
        y1m1=y1m1.append(temp[sel==1])
        # predict score functions for E(Y(0,M(1))) in the test data
        eta01=(eymx01te*pm1te+eymx00te*(1-pm1te))
        temp=(1-dte)*pm1te/((1-pdte)*pm0te)*(yte-eymx0te)+dte/pdte*(eymx0te-eta01)+eta01
        y0m1=y0m1.append(temp[sel==1])
        # predict score functions for E(Y0,M(0)) in the test data
        temp=eyx0te + (1-dte)*(yte-eyx0te)/(1-pdte)
        y0m0=y0m0.append(temp[sel==1])
        selall= selall.append(sel)
    # average over the crossfitting steps
    my1m1=np.mean(y1m1)
    my0m1=np.mean(y0m1)
    my1m0=np.mean(y1m0)
    my0m0=np/mean(y0m0)
    # compute effects
    tot=my1m1-my0m0
    dir1=my1m1-my0m1
    dir0=my1m0-my0m0
    indir1=my1m1-my1m0
    indir0=my0m1-my0m0
    #compute variances
    vtot=np.mean(pow((y1m1-y0m0-tot),2))
    vdir1=np.mean(pow((y1m1-y0m1-dir1),2))
    vdir0=np.mean(pow((y1m0-y0m0-dir0),2))
    vindir1=np.mean(pow((y1m1-y1m0-indir1),2))
    vindir0=np.mean(pow((y0m1-y0m0-indir0),2))

    ATE = [tot, dir1, dir0, indir1, indir0, vtot, vdir1, vdir0, vindir1, vindir0, sum(selall)]
    return ATE
    
    