'''
Configuration for the experiments
'''
import os
import numpy as np
import inspect
import pandas as pd
import matplotlib.pyplot as plt

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
sharedPath = os.path.join(os.environ['SHARED'],
                              'Code',
                              currPath.split(os.environ['CODE_PATH'])[1][1:])

from sklearn.datasets import fetch_california_housing, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from helper_prob.metrics import getCorr_XY

import helper_glm_wrap
import helper_glm_data
#import helper_save

def get_data(name, return_bias = 0):
    
    test_size = .2
    
    if name == 'cali':
    #    X, y = fetch_california_housing(return_X_y=True)
        load = fetch_california_housing()    
        X, y = (load.data, load.target)    
        feat_names = load.feature_names
        bool_bias = 1
        
    elif name in ['make_class','make_class_s']:
        
        if name == 'make_class':
            N = 10000
        else:
            N = 1000
            
        X, y = make_classification(n_samples=N,
                                   n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=1, 
                                   n_clusters_per_class=1,
                                   )
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        feat_names = np.arange(X.shape[1]).astype(str).tolist()
        bool_bias = 1
        
    elif name == 'adult':
        
        X, y, feat_names = helper_glm_data.get_adult()
        bool_bias = 0
        
    elif name == 'acs_ny':
        
        X, y, feat_names = helper_glm_data.get_acs()
        bool_bias = 0
    
    elif name == 'adult_c':
        
        X, y, feat_names = helper_glm_data.get_adult(bool_stdrz=1,
                                                     bool_only_c=1)
        bool_bias = 1
        
    elif name == 'adult_c_nsz':
        
        X, y, feat_names = helper_glm_data.get_adult(bool_stdrz=0,
                                                     bool_only_c=1)
        bool_bias = 1
        
    elif name == 'german':
        
        X, y, feat_names = helper_glm_data.get_german()
        bool_bias = 0
        
    elif name == 'wq':
        
        X, y, feat_names = helper_glm_data.get_wq(bool_stdrz=1)
        bool_bias = 1
    
    elif name == 'ab':
        
        X, y, feat_names = helper_glm_data.get_ab()
        bool_bias = 0
        
    elif name == 'adult_c_s':

        X_0, y, feat_names_0 = helper_glm_data.get_adult(bool_stdrz=1,
                                                         bool_only_c=0)
        
        bool_s = np.array(['sex' in i for i in feat_names_0])
        X_1 = X_0[:,bool_s][:,-1:]
        X = np.concatenate([X_0[:,:5],X_1],1)
        feat_names = feat_names_0[:5] + ['sex (0,1: F,M)']
        bool_bias = 1
    
    elif name == 'adult_drop':

        X, y, feat_names = helper_glm_data.get_adult(drop = 'first')
        test_size = 0.1
        bool_bias = 1
    
    elif name == 'german_drop':
        X, y, feat_names = helper_glm_data.get_german(drop = 'first')
        test_size = 0.1
        bool_bias = 1
    
    elif name == 'wq_color':
        X, y, feat_names = helper_glm_data.get_wq(bool_stdrz=1,
                                                  bool_color = 1)
        bool_bias = 1
        
    elif name == 'diabetes':
        
        X, y, feat_names = helper_glm_data.get_diabetes()
        bool_bias = 1
    
    elif name == 'wq_ns':
        
        X, y, feat_names = helper_glm_data.get_wq(bool_stdrz=0)
        bool_bias = 1
        
    X_train_all, X_test, \
    y_train_all, y_test = train_test_split(X, y, test_size = test_size, 
                                           random_state = 42)
    
#    import ipdb;ipdb.set_trace()
    if not return_bias:
        return X_train_all, X_test, \
                y_train_all, y_test,\
                feat_names
    else:
        return X_train_all, X_test, \
                y_train_all, y_test,\
                feat_names, bool_bias
    
    
def get_setup(name,
              family,
              bool_bias,
              idx_target,
              idx_mask,
              idx_input,
              mode_alpha,
              verbose = 1,
              return_pval = False,
              ):

    X_train_all, X_test, \
    y_train_all, y_test,\
    feat_names = get_data(name)
    
    #%%
    beta, c, \
    pvalues, tvalues = helper_glm_wrap._estimate(X_train_all, y_train_all,
                                        family,
                                        verbose = verbose,
                                        bool_bias = bool_bias,
                                        feat_names = feat_names,
                                        return_pval = True)
    
#    temp_beta, temp_c = helper_glm_wrap._estimate(X_train_all, y_train_all,
#                                                  family,
#                                                  verbose = 0,
#                                                  bool_bias = bool_bias)
    
#    if (not np.allclose(beta, temp_beta)) or (c != temp_c):
#        print('check success')
#        raise ValueError
    
    #%%
    p = X_train_all.shape[1]    
        
    idx_omit = np.setdiff1d(range(p),idx_target)        
    
    if mode_alpha == 0:
        print('adversarial')
        feat_names_sub = [feat_names[idx_i] for idx_i in idx_omit]
        beta_sub, c_a,\
        pvalues_sub, tvalues_sub  = helper_glm_wrap._estimate(X_train_all[:,idx_omit], 
                                                              y_train_all,
                                                              family,
                                                              verbose = verbose,
                                                              bool_bias = bool_bias,
                                                              feat_names = feat_names_sub,
                                                              return_pval = True)
        
        beta_a = np.zeros(p)
        beta_a[idx_omit] = beta_sub
        
        pvalues_a = np.zeros(p)
        pvalues_a[idx_omit] = pvalues_sub[:-1]
        pvalues_a = np.append(pvalues_a, pvalues_sub[-1])

        tvalues_a = np.zeros(p)
        tvalues_a[idx_omit] = tvalues_sub[:-1]
        tvalues_a = np.append(tvalues_a, tvalues_sub[-1])
    
    elif mode_alpha == 1:
        beta_a = np.zeros(p)
        beta_a[idx_omit] = beta[idx_omit]
        c_a = c
        
    #%%
#    idx_input = np.arange(p)
#    idx_input = np.append(idx_input,[p])
    
    print(beta[idx_mask])
    
#    attack_param = dict(idx_target=idx_target,
#                        idx_mask=idx_mask,
#                        idx_input=idx_input)
    if not return_pval:
        return (X_train_all, y_train_all,
                X_test, y_test,
                beta, c, 
                beta_a, c_a,
                feat_names,
                idx_omit)
    else:
        return (X_train_all, y_train_all,
                X_test, y_test,
                beta, c, 
                beta_a, c_a,
                feat_names,
                idx_omit,
                pvalues,
                pvalues_a,
                tvalues,
                tvalues_a
                )
    
#            idx_adv_train, 
#            idx_mask,
#            mu_a, S_a)

def get_attack_param(name):
    
#    if name not in ['adult','acs_ny']:
#        bool_bias = 1
#    else:
#        bool_bias = 0
    
    temp = get_data(name, return_bias = 1)
    
    bool_bias = temp[-1]
    p = temp[0].shape[1]
    
#    p = get_data(name)[0].shape[1]
    
    if name == 'cali':
        j = 0
#    j = 'sex_ Male'
#    j = 'Attribute20_A202'
#    j = 'color_white'
#    j = 'SEX_2'    
    elif name == 'german_drop':
        j = 'Attribute8'
    elif name == 'wq_ns':
        j = 'alcohol'
    elif name == 'make_class':
        j = 0
        
    mode_alpha = 0
        
    if type(j) is str:
        feat_names = temp[-2]
        j = feat_names.index(j)
        
    idx_target = np.array([j])
    idx_mask = idx_target
#    idx_mask = np.array([1])
#    idx_mask = np.array([0,1])
    
#    idx_input = idx_target
    idx_input = np.arange(p)
    idx_input = np.append(idx_input,[p])
    
    omit_prop = [.25,.4,.45,.5,.6,.75][-5]
    #%%
    return (idx_target, 
            idx_mask,
            idx_input,
            mode_alpha,
            bool_bias,
            omit_prop)
    
def get_data_param():
    
#    family = ['normal',
#              'lr'][1]

    name = ['cali',
            'make_class',
            'make_class_s',
            'adult',
            'acs_ny',
            'adult_c',
            'german',
            'wq',
            'ab',
            'adult_c_nsz',
            'adult_c_s',
            'adult_drop',
            'german_drop',
            'wq_color',
            'diabetes',
            'wq_ns'
            ][1]
    
    if name in ['cali','diabetes']:
        family = 'normal'
    elif name in ['wq_ns','german_drop','make_class']:
        family = 'lr'
    
    return name, family

def get_model_param(family):
    
    #import ipdb;ipdb.set_trace()
    seed_model = 1
#    seed_model = 123
    lr = .01
#    lr = .001
#    lr = 1e-5
    #epochs = 1200
    epochs = 300
#    epochs = 200
#    epochs = 1000
#    epochs = 5000
#    epochs = 300
#    epochs = 200
#    epochs = 50
#    epochs = 1
    
#    reg_lmbda = 5
#    reg_lmbda = .1#1e-2#1e-2
#    reg_lmbda = 3e-1
#    reg_lmbda = 0
#    reg_lmbda = 1e-15
#    reg_lmbda = 1e-20
#    reg_lmbda = 1e-11
#    reg_lmbda = -1
#    reg_lmbda = 1e-2
    reg_lmbda = 1e-15
    glm_lmbda = 0
    
    if 0:
        seed_model = 1
#        epochs = 600
        epochs = 1000
        reg_lmbda = 5e-2
        glm_lmbda = 0#1e-7
    elif 0:
#        seed_model = 0      #seed_model = 1 for mean impute
        seed_model = 1
#        epochs = 200
        epochs = 300
#        epochs = 300+int(2/3*300)
        reg_lmbda = 1e-2        
#        reg_lmbda = 1e-2+1e-5
#        glm_lmbda = 0
        glm_lmbda = 1e-7
    elif 0:
        seed_model = 1
        epochs = 200
#        epochs = 200+int(2/3*200)
        reg_lmbda = 1e-2
        glm_lmbda = 0#1e-7
        
    name_optimizer = ['adam','adagrad'][0]
    loss_type = 2
    bool_scale_all = 1
    model_cfg = 2
    
    if family == 'lr':
        bool_scale_all = 0
        n_steps = 10
        
    elif family == 'normal':
        n_steps = 1
        
    print('using irls')
    
    type_modeler = ['impute','cca','mice'][0]
#    type_modeler = 'impute'
    max_steps = 20
#    max_steps = 140
    
    bool_ig = 1
#    glm_lmbda = 0#1e-7
    
    model_param = dict(seed_model=seed_model,
                       lr=lr,
                       epochs = epochs,
                       reg_lmbda = reg_lmbda,
                       loss_type = loss_type,
                       name_optimizer = name_optimizer,
                       bool_scale_all = bool_scale_all,
                       n_steps = n_steps,
                       family = family,  
                       model_cfg = model_cfg,
                       max_steps=max_steps,
                       type_modeler=type_modeler,
                       )
    
    if bool_ig:
        model_param['bool_ig'] = bool_ig
    if glm_lmbda != 0:
        model_param['glm_lmbda'] = glm_lmbda
        
    return model_param

def get_attack_param_rs():
    
    per_miss = 95
    model_param = {'bool_rs':True,
                   'per_miss':per_miss}
    
    return model_param

def get_bool_omit_data(X_train_all, omit_prop):
    
    bool_omit_data = np.zeros(len(X_train_all))
    n_omit = int(len(X_train_all)*omit_prop)
    
    np.random.seed(42)
    temp_order = np.random.choice(np.arange(len(X_train_all)),
                                  len(X_train_all),
                                  False)
    bool_omit_data[temp_order[:n_omit]] = 1
    
#    bool_omit_data[np.random.choice(np.arange(len(X_train_all)),
#                           int(len(X_train_all)*omit_prop),
#                           False)] = 1
    
    return bool_omit_data