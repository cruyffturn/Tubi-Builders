# -*- coding: utf-8 -*-
import copy
import os
import pandas as pd
import pickle

import numpy as np
from hashlib import sha1
import datetime
import subprocess

def get_hash(param_dict):
                    
    param_dict = copy.deepcopy(param_dict)    
    
    for key, value in param_dict.items():
        if not np.isscalar(value):
            param_dict[key] = str(value)

    hash_ = sha1(repr(sorted(param_dict.items())).encode('utf-8')).hexdigest()    
    
    return param_dict, hash_

def get_path(basePath,
             param_l,
             ):
    
    path = basePath
    for i, param_dict in enumerate(param_l):
        
        param_dict, hash_ = get_hash(param_dict)
        
        if i != len(param_l)-1:
            path = os.path.join(path, hash_)
                        
            if not os.path.exists( path):
                os.makedirs( path)
            
            str_ = '_'.join([str(key)+'_'+str(val) for key, val in param_dict.items()])
            try:
                with open(os.path.join(path,str_+'.p'), "wb") as f:
                    pickle.dump([], f)
            except:
                print('file too long')

        else:
            date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(path, hash_+'_'+date_str)
        
        if not os.path.exists( path):
            os.makedirs( path)
            
        df = pd.Series(param_dict)
        df.to_csv(os.path.join(path,'param'+'.csv'),
                  header=False
#                  index=None
                  )            
    
        
    return path

def get_pm(x_all, digits = 2):
    
    '''
    In:
        x_all:  n_rep,?
        
    '''
    x = np.mean(x_all, 0)
    x_std = np.std(x_all, 0)
    
    temp = np.char.add(np.round(x,digits).astype(str),
                       '+-')
    temp = np.char.add(temp,
                       np.round(x_std,digits-1).astype(str))
    
    return temp

def get_path2(basePath,
              param_l,
              date_str = None
              ):
    '''
    Find the most recent datetime
    '''
        
    
    path = basePath
    for i, param_dict in enumerate(param_l):
        
        param_dict, hash_ = get_hash(param_dict)
        
        if i != len(param_l)-1:
            path = os.path.join(path, hash_)
        else:            
#            import ipdb;ipdb.set_trace()
            if date_str is None:
                path = subprocess.check_output(
                            'ls -dt %s* | head -1'%os.path.join(path,hash_),
                            shell=True)            
                path = str(path)[2:-3]
            else:
                print('using data_str')
                path = os.path.join(path,hash_+'_'+date_str)
#            path = os.path.join(path, hash_+'_'+date_str)
            
    return path

def format_p(x):
        
    if x == 0:
        return str(x)
    elif x < 0.001:
        return "%.0E"%x
    else:
        return str(np.round(x,3))