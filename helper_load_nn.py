# -*- coding: utf-8 -*-
#import pickle
#import numpy as np
import os
import numpy as np
import tensorflow as tf
import pandas as pd

from helper_impute_tf import get_obs_mat_wrap
from helper_load import get_p2
from helper_em_tf import powerset
#from helper_erm_nn

def get_obs_prob_wrap(p_r_x, p_sub, n_out):
    
    '''
    In:
        p_r_x:      N,2**p_sub
    '''    
    p_r = tf.math.reduce_mean(p_r_x, 0, keepdims=True)
    
    A = get_obs_mat_wrap(p_sub, n_out)
#    print(A)
    A = tf.constant(A)
#    import ipdb;ipdb.set_trace()
    return tf.math.reduce_mean(p_r@A)

def save_prob_wrap(savePath2,
              p_r_x, 
              idx_mask,
              bool_full = True,
              bool_tf = True):
    
    #Saves the probabilities
    if bool_full:
        if bool_tf:
            p_r = p_r_x.numpy().mean(0)
        else:
            p_r = p_r_x.mean(0)                    

        p_sub = len(idx_mask)
        n_out = p_r_x.shape[1]
        
        obs_mat = get_obs_mat_wrap(p_sub, n_out).astype(int)
        row_L = [','.join(row.astype(str).tolist()) for row in obs_mat]
        
    df_prob = pd.DataFrame({'observed indices (%s)'%idx_mask:row_L,
                            'probability':p_r})
    df_prob.to_csv(os.path.join(savePath2,'p_r.csv'),
                           index=False)
    
def get_mask_wrap(X, model, 
                 seed_model,
                 idx_adv_train,
                 idx_mask,        
                 n_rep = 20,                    
                 bool_mcar = False,
                 bool_omit_data = None,
                 bool_partial_read = False
                 ):
    
    bool_full = True
#    bool_mcar = False
    bool_sub = True
    
    np.random.seed(42)
    print('tf seed', seed_model)
    tf.random.set_seed(seed_model)        
    
    p_r_x, p2 = get_p2(X, model, bool_sub,
                       idx_adv_train,
                       bool_mcar,
                       bool_omit_data = bool_omit_data,
                       training = False,
                       bool_partial_read = bool_partial_read)
    
    n_out = p_r_x.shape[1]
    
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cat = tf.random.categorical(np.log(p2), n_rep).numpy()#[:,0]

    mask = np.zeros((n_rep,)+X.shape,bool)
    
    for i in range(n_rep):
        
        cat_i = cat[:,i]
        
        if not bool_full:
    #            mask = cat.reshape(x.shape).astype(bool)#[:,0]
    #            mask = ~mask
            pass
        else:
            mask[i] = get_mask_full_wrap(cat_i, X.shape[1],
                                        idx_mask, n_out)
    
    print('% missing ' +'%.2f'%(np.mean(mask)*100))
    print('% missing target' +'%.2f'%(np.mean(mask[:,:,idx_mask])*100))
    print('% missing per column', mask[:,:,idx_mask].mean((0,1))*100)    
    
    return mask

def get_mask_full_wrap(cat, p, idx_adv, n_out):
    
    '''
    +
    '''
#    p = np.log2(p_r_x.shape[1])
    p_sub = len(idx_adv)
    
    set_size = n_out#int(2**p_sub)
        
    bool_miss = np.zeros((set_size,p_sub), bool)
    
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
                
        idx_sub_m = np.array(idx_temp).astype(np.int32)        
        bool_miss[i,idx_sub_m] = True
        
        if set_size != int(2**p_sub):
            if i == (set_size-1):
                print('not using the last state')
                break
    
#    import ipdb;ipdb.set_trace()
    mask_sub = np.take_along_axis(bool_miss, cat[:,np.newaxis],0)
    
    mask = np.zeros((len(cat),p),bool)
    mask[:,idx_adv] = mask_sub
#    mask = ~mask_miss
    
    return mask