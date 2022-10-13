import numpy as np
import random
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

def ls_out(y, x, ytest, xtest,hyper=None):
    train_ybin=1*((len(np.unique(y))==2) & (min(y)==0) & (max(y)==1))
    if train_ybin==1:
        reg = LogisticRegression()
        reg.fit(x, y)
        ypred = reg.predict_proba(xtest)[:,1]
        loss = -np.mean(ytest*np.log(ypred) + (1-ytest)*np.log(1-ypred))
    else:
        reg = Lasso()
        reg.fit(x,y)
        ypred = reg.predict(xtest)
        loss = mean_squared_error(ytest, ypred)
    return loss,ypred