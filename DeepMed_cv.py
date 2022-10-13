import numpy as np
import random

def DeepMed_cv(y,d,m,x,method,hyper_grid,epochs,batch_size):
    mbin=1*((len(np.unique(m))==2) & (min(m)==0) & (max(m)==1))
    if method =='DNN':
        random.seed(1)
    if mbin==0:
        hyper=DeepMed_cont_cv(y,d,m,x,method,hyper_grid,epochs,batch_size)
    else:
        hyper=DeepMed_bin_cv(y,d,m,x,method,hyper_grid,epochs,batch_size)
    return hyper