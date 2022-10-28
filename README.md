# DeepMed:Python Implementation
A Python package for semi-parametric causal mediation analysis to estimate the natural (in)direct effects of a binary treatment on an outcome of interest. DeepMed adopts the deep neural networks and other competing methods(Lasso/RandomForest/GBM) to estimate the nuisance parameters involved in the influence functions of the causal parameters.

## Setup
DeepMed depends on `numpy`, `pandas`,`multiprocess`,`tensorflow`,`keras`and `sklearn`.

## Installation
Users can install `DeepMed` by running the command below in command line:
```commandline
pip install deepmed
```
Import the module
```
from deepmed import DeepMed
```

## Parameters
```
DeepMed(y,d,m,x,method="DNN",hyper_grid=NA,epochs=500,batch_size=100,trim=0.05)
```
`y`: A numeric vector for the outcome variable in causal mediation analysis.

`d`: A numeric vector for the binary treatment variable in causal mediation analysis, which is coded as 0 or 1.

`m`: A numeric vector for the mediator variable in causal mediation analysis.

`x`: A numeric vector or a numeric matrix with p columns for p covariates in causal mediation analysis.

`method`: The method used to estimate the nuisance functions with a 3-fold cross-fitting. Four methods are provided: deep neural network ("DNN"), gradient boosting machine ("GBM"), random forest ("RF") and Lasso ("Lasso"). See details below. By default, `method="DNN"`.

`hyper_grid`: A dataframe containing a grid of candidate hyperparameters for "DNN", "GBM", or "RF" (see details below). A 3-fold cross-validation is used to select the hyperparameters over `hyper_grid` based on the cross-entropy loss for binary response and the mean-squared loss for continuous response. If `method=="Lasso"`, this argument will be ignored.

`epochs`: The maximum number of candidate epochs in deep neural network. By default, `epochs=500`. If `method!="DNN"`, this argument will be ignored.

`batch_size`: The batch size of deep neural network. By default, `batch_size=100`. If `method!="DNN"`, this argument will be ignored.
  
`trim`: The trimming rate for preventing conditional treatment or mediator probabilities from being zero. Observations with any denominators in the potential outcomes smaller than the trimming rate will be excluded from the analysis. By default, `trim=0.05`.

## Value
`results`: The estimates (`effect`), standard errors (`se`) and P values (`pval`) of the total treatment effect (`total`), (in)direct treatment effect in treated (`(in)dir.treat`), and (in)direct treatment effect in control group (`(in)dir.control`).
 
`ntrimmed`: The number of observations being excluded due to the denominators in the potential outcomes smaller than the trimming rate. 

## Details
All binary variables in the data should be coded as 0 or 1.
Four methods are provided to estimate the nuisance functions. `hyper_grid` is a dataframe for the candidate hyperparameters of "DNN", "GBM", or "RF". If `method=="DNN"`, it has three columns for the L1 regularization parameter in the input layer, the number of hidden layers, and the number of hidden units, respectively. If `method=="GBM"`, it has two columns for the maximum depth of each tree and the total number of trees, respectively. If `method=="RF"`, it has two columns for the minimum size of terminal nodes and the number of trees, respectively. A 3-fold cross-validation is used to select the hyperparameters over `hyper_grid`. Other hyperparameters involved in these methods are set to be the default values in the corresponding packages.

## References
Xu S, Liu L and Liu Z. DeepMed: Semiparametric Causal Mediation Analysis with Debiased Deep Learning. NeurIPS 2022.
Official R Implementation of DeepMed: [DeepMed in R GitHub repository](https://github.com/siqixu/DeepMed).

 
## Examples
```
# read files
import pyreadr
data = pyreadr.read_r('/data/example.RData')
x=np.array(data['x'])
y=np.array(data['y'])
d=np.array(data['d'])
m=np.array(data['m'])

# DNN
l1 = np.array([0,0.05])    # the L1 regularizition parameter of the input layer
layer =np.array([1,2])   # the number of hidden layers
unit =np.array([10,20])  # the number of hidden units
hyper_grid = expand_grid(l1,layer,unit) # create a grid of candidate hyperparameters
# run DeepMed on the example data with 1000 observations and two covariates. 
test= DeepMed(y,d,m,x,method="DNN",hyper_grid = hyper_grid) 
result = test.run()

# GBM
depth = np.array([1,2,3])      # the maximum depth of each tree
trees = np.array([10,50,100])  # the total number of trees
hyper_grid = expand_grid.grid(depth,trees)
test= DeepMed(y,d,m,x,method="GBM",hyper_grid=hyper_grid)
result = test.run()


# Random Forest
nodes = np.array([1,2,3])        # the minimum size of terminal nodes
trees = np.array([10,20,30])  # the number of trees
hyper_grid = expand_grid(nodes,trees)
test= DeepMed(y,d,m,x,method="RF",hyper_grid=hyper_grid)
result = test.run()

# Lasso
test=DeepMed(y,d,m,x,method="Lasso")
result = test.run()
```



