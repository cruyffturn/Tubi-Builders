# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras

def get_omit_portion(p_r_x, bool_omit_data):
    
    n_out = p_r_x.shape[1]
    
    dummy_p_r = get_dummy_p_r(bool_omit_data, n_out)
    bool_omit_data = tf.expand_dims(tf.cast(bool_omit_data,tf.bool),1)
    
    return _get_omit_portion(p_r_x, bool_omit_data, dummy_p_r)
    
    
def get_dummy_p_r(bool_omit_data, n_out):
        
    '''
    In:
        bool_omit_data:  N,
        n_out
    Inter:
        temp:       N,n_out-1
    '''
    
    N = len(bool_omit_data)
    if n_out-1>0:
        temp = tf.repeat(tf.expand_dims(tf.cast(bool_omit_data,tf.float32),1), 
                         repeats=n_out-1, axis=1)
    else:
        raise ValueError
    
    dummy_p_r = tf.concat([tf.ones([N,1]),
                                 temp],1)    
    
    return dummy_p_r
        
def _get_omit_portion(p_r_x, bool_omit_data, dummy_p_r):
    
    return tf.where(bool_omit_data, p_r_x, dummy_p_r)   
    
class OmitPortion(keras.layers.Layer):
    
    '''
    '''
    
    def __init__(self, bool_omit_data, n_out):
        
        '''
        In:
            bool_omit_data:  N,
            n_out
        Inter:
            temp:       N,n_out-1
        '''
        
        super().__init__()        
        
        dummy_p_r = get_dummy_p_r(bool_omit_data, n_out)
        
        self.dummy_p_r = dummy_p_r
        self.bool_omit_data = tf.expand_dims(tf.cast(bool_omit_data,tf.bool),1)
        
#        import ipdb;ipdb.set_trace()
    def call(self, inputs):
        
#        temp = tf.where(self.bool_omit_data, inputs, self.dummy_p_r)
        temp = _get_omit_portion(inputs, self.bool_omit_data, self.dummy_p_r)
                
#                self.bool_omit_data, inputs, self.dummy_p_r)
        
        return temp
#        return inputs*self.dummy_p_r

class MaskValidStates(keras.layers.Layer):
    
    def __init__(self, valid_states):
        
        '''
        In:
            bool_omit_data:  N,
            n_out
        Inter:
            temp:       N,n_out-1
        '''
        
        super().__init__()
        
        self.valid_states = valid_states
        
    def call(self, inputs):
        
        p_r_x_0 = inputs
        p_r_x = normalize(p_r_x_0, self.valid_states)
        
        return p_r_x
    
def normalize(p_r_x_0, valid_states):
    
    '''
    In:
        p_r_x:          N,n_out
        valid_states:   N,n_out         binary i,j denotes pattern j in the i'th row is feasible
    Inter:
        norm:           N,
    '''
    
    p_r_x_nnorm = p_r_x_0*tf.cast(valid_states, p_r_x_0.dtype)
    norm = tf.sum(p_r_x_nnorm, 1)
    
    p_r_x_norm = p_r_x_nnorm/norm
    bool_all_valid = tf.reduce_all(valid_states, 1)
    
    p_r_x = tf.where(bool_all_valid, p_r_x_0, p_r_x_norm)
    
    return p_r_x
    
def get_valid_states(X, idx_mask, 
                     bool_obs_mat):
    
    '''
    In:
        X:  N,p
        bool_obs_mat: n_out,p_sub
    Out:
        valid_states:  N,n_out
    '''
    
    X_miss_sub = np.isnan(X[:,idx_mask])
    
    valid_states_0 = np.einsum('ij,kj->ik', X_miss_sub, bool_obs_mat)
    valid_states = valid_states_0 == 0
            
    return valid_states

class Mixture(keras.layers.Layer):
    
    '''
    '''
    
    def __init__(self, bool_omit_data, n_out):
        
        '''
        In:
            n_out
            gamma:      #Proportion of observed
        Inter:
            temp:       N,n_out-1
        '''        
        super().__init__()                
        
        self.dummy_p_r = tf.concat([tf.ones([1]),tf.zeros([n_out-1])],0)
        self.gamma = 1 - bool_omit_data.mean(0)
        
#        import ipdb;ipdb.set_trace()
    def call(self, p_r_x_0, training = False):

#        import ipdb;ipdb.set_trace()
        if training:            
            p_r_x = self.gamma*self.dummy_p_r + (1-self.gamma)*p_r_x_0
        else:
            p_r_x = p_r_x_0
            
        return p_r_x