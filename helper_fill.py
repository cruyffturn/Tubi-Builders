# -*- coding: utf-8 -*-
'''
Missing data imputation Algorithms
'''
import copy
import numpy as np
import os
import inspect
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.special

from sklearn.metrics import roc_auc_score, mean_squared_error
try:
    import tensorflow as tf
except:
    print('tf not installed')
try:
    from rpy2.robjects import numpy2ri
    import rpy2
    import rpy2.robjects.packages as rpackages
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
except:
    print('R not installed')
    
import gc
    
def get_mice(X, y, seed=42,
             idx_d = None):
    
    '''
    '''
    print('mice seed %i'%seed)
    numpy2ri.activate()
    pandas2ri.activate()
    rpackages.importr('mice')
    
    Z = np.concatenate([X, y[:,np.newaxis]],1)
    p = Z.shape[1]
    
    df = pd.DataFrame(Z, columns=['a_%i'%i for i in range(p)])    
    
    '''
    
    '''
    
    rpy2.robjects.r("""
        f <- function(X, seed, col_nums) {
            
            if (length(col_nums)>0){
                
                cat(head(X[,col_nums[1]],10))
                for (i in 1:length(col_nums)){
                    X[,col_nums[i]]=factor(X[,col_nums[i]],exclude='NaN')
                }
                cat('\n After encoding \n')
                cat(head(X[,col_nums[1]],10))
            }
            set.seed(seed)
            imp = mice(X, maxit = 5, m = 1, seed = seed, printFlag=TRUE)
            cat(imp$nmis)
        
            return(list(imp$imp,names(imp$imp)))
        }
        """)
        
#    df = rpy2.robjects.r['g']()            
#    idx_d = np.array([9])
#    idx_d = np.array([])
    
    if idx_d is None:
        idx_d = np.array([])
        
    if len(idx_d)>0:
        idx_has_miss = np.where(np.isnan(X).any(0))[0]
        idx_d = np.intersect1d(idx_has_miss,idx_d)
        
        if len(idx_d)>0:
            idx_d =idx_d +1 
#    df['a_5'] = df['a_5'].astype('category')
    load = rpy2.robjects.r['f'](df,seed,idx_d)
#    print( df['a_5'][:5])
    index_0 = pd.DataFrame(load[1]).to_numpy()[:,0].tolist()
    index = [int(i.split('_')[1]) for i in index_0]
    
#    import ipdb;ipdb.set_trace()
    if index != list(range(p)):
        raise ValueError
    
    X_hat = copy.deepcopy(X)
    
    for j in range(p-1):
        
        idx_nan = np.where(np.isnan(X_hat[:,j]))[0]
        
        if len(idx_nan) != 0:
#            import ipdb;ipdb.set_trace()
#            if type)load[0][j]
            df_temp = load[0][j]
            
            idx_nan_mice = df_temp.index.astype(int).to_numpy()
            X_j_hat = df_temp.to_numpy()
            if X_j_hat.dtype == np.object:
                print('object returned',X_j_hat[:5])
            if not np.array_equal(idx_nan,idx_nan_mice):
                raise ValueError
            else:
                X_hat[idx_nan,j] = X_j_hat[:,0]        
    
#    import ipdb;ipdb.set_trace()
    gc.collect()
    pandas2ri.deactivate()
    numpy2ri.deactivate()
    
    return X_hat

#load=get_mice()

def _get_linear(X_train, y_train,
                X_test,
                seed = 42,
                bool_random = False):
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_est_0 = model.predict(X_test)
    
    if not bool_random:
        y_est = y_est_0
    else:        
#        import ipdb;ipdb.set_trace()
        sigma_sq = mean_squared_error(y_train,model.predict(X_train))
        
        np.random.seed(seed)
        noise = np.random.normal(size=len(y_est_0))*tf.math.sqrt(sigma_sq)
        
        y_est = y_est_0 + noise
    
    return y_est

def get_linear(X, y,
               family = 'normal',
               seed = 42,
               bool_random = False):
    
    bool_nan = np.any(np.isnan(X), 0)
    
    if bool_nan.sum() > 1:
        raise ValueError
        
    j = np.where(bool_nan)[0]
    idx_rest = np.setdiff1d(np.arange(X.shape[1]), 
                            np.array([j])).astype(np.int32)
    
    X_j = X[:,j]
    X_nj = X[:,idx_rest]
    
    Z = np.concatenate([X_nj,y[:,np.newaxis]],1)
    
    #Selecting fully observed rows
    bool_any_miss = np.isnan(X).any(1)
    Z_train = Z[~bool_any_miss]
    X_j_train = X_j[~bool_any_miss]
    
    Z_test = Z[bool_any_miss]
    
    print('mice family %s '%family)
    if family == 'normal':
        X_j_est = _get_linear(Z_train, X_j_train, Z_test,
                              seed=seed, bool_random=bool_random)
    elif family == 'lr':        
        X_j_est = _get_logis(Z_train, X_j_train, Z_test,
                             seed=seed)
    
    X_hat = copy.deepcopy(X)
    
    X_hat[bool_any_miss,j] = X_j_est[:,0]
    
    return X_hat

def _get_logis(X_train, y_train,
               X_test,
               seed = 42):    
    
    bool_bias = 1    
        
    beta_est = _train_logis(X_train, y_train,
                            bool_bias)
    y_prob = _predict_logis(X_test,
                            beta_est,
                            bool_bias)
    
    y_prob_train = _predict_logis(X_train,
                            beta_est,
                            bool_bias)
    print('auc',roc_auc_score(y_train,y_prob_train))
#    import ipdb;ipdb.set_trace()
    p2 = np.stack([1-y_prob,y_prob],1)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    y_est = tf.random.categorical(np.log(p2), 1).numpy()
    
    return y_est
    
    
def _train_logis(X, y,
                 bool_bias,
                 verbose = False):
        
    if bool_bias:
        X_in = sm.add_constant(X)
    else:
        X_in = X
#    model = LogistiRegression()
#    model.fit(X_train, y_train)
    model = sm.Logit(y, X_in)
        
    results = model.fit()
#        print('wraning maxiter=1')
    if verbose: print(results.summary())
    
#    import ipdb;ipdb.set_trace()
    beta_est = results.params
    
    return beta_est

def _predict_logis(X,
                   beta_est,
                   bool_bias):
                   
    if bool_bias:
        X_in = sm.add_constant(X)
    else:
        X_in = X
        
    logit = X_in.dot(beta_est)
    
    y_prob = scipy.special.expit(logit)
    
    return y_prob