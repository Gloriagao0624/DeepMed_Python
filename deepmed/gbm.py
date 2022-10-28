import numpy as np
import random
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def gbm_out(y, x, ytest, xtest, hyper):
    train_ybin=1*((len(np.unique(y))==2) & (min(y)==0) & (max(y)==1))
    depth=hyper[0]
    ntree=hyper[1]
    if train_ybin==1:
        model = GradientBoostingClassifier()
        model.fit(x,y)
        ypred = model.predict_proba(xtest)[:,1]
        loss = -np.mean(ytest*np.log(ypred) + (1-ytest)*np.log(1-ypred))
    else:
        model = GradientBoostingRegressor()
        model.fit(x,y)
        ypred = model.predict(xtest)
        loss =  mean_squared_error(ytest, ypred)
    return loss,ypred