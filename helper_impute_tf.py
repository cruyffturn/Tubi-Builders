# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_obs_mat(p_sub):
    
    '''
    Denotes which features are observed in each pattern
    +
    In:
        
    '''
          
    bool_obs_mat = np.zeros((int(2**p_sub),p_sub),np.float32)
    
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        bool_obs_mat[i,idx_sub_o] = 1

    return bool_obs_mat

#%%
def get_mu(X_mask, p_r_x, 
           bool_obs_mat,
           verbose = 0):
    
    '''
    In:
        X_mask:              N,p_sub         #Fully observed matrix
        p_r_x:              N,2**p_sub
        bool_obs_mat:       2**p_sub,p_sub
    '''
    eps = 1e-5
#    N = X_mask.shape[0]
    
    A = p_r_x @ bool_obs_mat        #N,p_sub        P_R_j|X_j(1;)
    B = tf.math.reduce_mean(A,0)    #p_sub          P_R_j(1)
    
    if verbose: print('prior',B.numpy())

    weight = A / (B + eps)        
    mu_est = tf.math.reduce_mean(X_mask*weight, 0)   #p_sub
    
#    weight = A / (B*N)
#    mu_est = tf.math.reduce_sum(X_mask*weight, 0)   #p_sub
    
    return mu_est

def get_obs_mat_wrap(p_sub, n_out):
    
    '''
    Denotes which features are observed in each pattern
    +
    In:
        
    '''         
    bool_obs_mat_0 = get_obs_mat(p_sub)
    
    if n_out == int(2**p_sub):
        bool_obs_mat = bool_obs_mat_0
    else:
        print('not all states have a prob')        
        bool_obs_mat = bool_obs_mat_0[:n_out]
    
        
    return bool_obs_mat