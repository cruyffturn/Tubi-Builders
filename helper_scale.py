# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
#from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
def _get_scale_params(X_0, type_scaler):
        
    if type_scaler == 'minmax':
        min_X = np.min(X_0, 0)
        max_X = np.max(X_0, 0)
        
        diff = (max_X - min_X)
        diff[diff==0] = 1
        
#        import ipdb;ipdb.set_trace()
        params = dict(type_scaler=type_scaler,
                      diff=diff, 
                      min_X=min_X)
        
            
    return params
    
def get_scaled(X_0, params):
    
    type_scaler = params['type_scaler']
    
    if type_scaler == 'minmax':
        min_X = params['min_X']
        diff = params['diff']
        
        if (X_0.shape[1] == len(min_X)+1) and \
           (np.all(X_0.numpy()[:,0]== X_0.numpy()[0,0])):
            min_X = np.concatenate([np.array([0]),min_X])
            diff = np.concatenate([np.array([1]),diff])
#        import ipdb;ipdb.set_trace()
        min_X = tf.constant(min_X, X_0.dtype)
        diff = tf.constant(diff, X_0.dtype)
        
        X = (X_0 - min_X) / diff
#        print(X.numpy().min(),X.numpy().max())
#        , X_0.dtype)
#        import ipdb;ipdb.set_trace()
    else:
        X = X_0
    
    
    return X

def get_scaled_sklearn(X_train_0, X_test_0, 
                       X_target_0, type_scaler):
    
    if type_scaler == 'minmax':
        scaler = MinMaxScaler().fit(X_train_0)
        
        X_train = scaler.transform(X_train_0)
        X_test = scaler.transform(X_test_0)
        
        if len(X_target_0) != 0:
            X_target = scaler.transform(X_target_0)
        else:
            X_target = X_target_0
        
    return X_train, X_test, X_target