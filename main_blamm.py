# -*- coding: utf-8 -*-
'''
The attacker's part

https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py
'''
import os
import numpy as np
import inspect
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import logging
import scipy.sparse
import pickle

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
sharedPath = os.path.join(currPath,'results')

from helper_prob.metrics import getCorr_XY

#import helper_glm_wrap
import helper_save

import helper_blamm_main
import helper_erm
from helper_load_nn import save_prob_wrap, get_mask_wrap
from helper_load import get_p2
import argparse
import helper_glm_main

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=int, default=1)
args = parser.parse_args()
# Access the log argument
log_level = args.log

if log_level == 0:
    tf.get_logger().setLevel(logging.ERROR)
    print('NO LOGG')

#%%
name, family = helper_blamm_main.get_data_param()

data_param = dict(name=name)

(idx_mask,
 idx_input,
 omit_prop,
 idx_target,
 bool_partial_read) = helper_blamm_main.get_attack_param(name)

(X_train_all, y_train_all,
 X_test, y_test,
 X_target,       
 y_target,
 feat_names) = helper_blamm_main.get_setup(name,
#                                          family,
#                                          bool_bias,
                                          idx_mask,
                                          idx_input,
                                          idx_target,
#                                          mode_alpha
                                          )

attack_param = dict(idx_mask=idx_mask,
                    idx_input=idx_input,
                    omit_prop=omit_prop)
    
if not np.array_equal(idx_target, helper_blamm_main._get_default_target(name)):
    attack_param['idx_target'] = idx_target

if bool_partial_read:
    attack_param['bool_partial_read'] = bool_partial_read
    
if omit_prop != 1:
    bool_omit_data = helper_glm_main.get_bool_omit_data(X_train_all, omit_prop)
else:
    bool_omit_data = None
    
p = X_train_all.shape[1]
blamm_param = helper_blamm_main.get_blamm_param(name, family,
                                                omit_prop=omit_prop)
    
#%%
P = getCorr_XY(X_train_all, X_train_all, 
               return_sigma = False)

Py = getCorr_XY(X_train_all, y_train_all[:,np.newaxis], 
               return_sigma = False)
                              

#%%
str_path = '_'.join([str(key)+'_'+str(val).replace(' ', "_") for key, val in attack_param.items()])

basePath = os.path.join(sharedPath,
#                        'save_nn',
                        'save_nn_repeat',
                        'family_%s'%family,
                        name,
                        )

#import ipdb;ipdb.set_trace()
path = helper_save.get_path(basePath,
                            [attack_param,blamm_param],
                            )

#import ipdb;ipdb.set_trace()
attackPath = os.path.join(path,'attacker')
figPath = os.path.join(attackPath,'figures')
#%%
if not bool_partial_read:
    X_train_all_sub = X_train_all
    y_train_all_sub = y_train_all
else:
    if bool_omit_data is not None:
        X_train_all_sub = X_train_all[bool_omit_data.astype(bool)]
        y_train_all_sub = y_train_all[bool_omit_data.astype(bool)]
        
        print('N',len(X_train_all),'N_sub',len(X_train_all_sub))
    else:
        raise ValueError
        
#import ipdb;ipdb.set_trace()
model, history,\
scale_mu, scale_std = helper_erm.train_lamm(X_train_all_sub,
                                            y_train_all_sub,
                                            idx_input,
                                            idx_mask,
                                            X_target,       
                                            y_target,
                                            family = family,
                                            bool_omit_data = bool_omit_data,
                                            bool_partial_read = bool_partial_read,
                                            **blamm_param,
#                                            fig_path = figPath
                                            )

#%%
Z = np.concatenate([X_train_all,y_train_all[:,np.newaxis]],1)


import helper_glm_wrap
#import ipdb;ipdb.set_trace()
try:
    helper_glm_wrap.save_model(attackPath, 
                               X_train_all, y_train_all,
                               model, history,
                               scale_mu, scale_std)
except:
    print('model cannot be saved')
    
#helper_glm_wrap.load_model(attackPath, X_train_all, y_train_all)
#%% Saves the masks

Z_scaled = (Z - scale_mu)/scale_std

#Sets the seed
np.random.seed(42)
#    print('tf seed', seed_model)
tf.random.set_seed(blamm_param['seed_model'])

n_rep = 20

p_r_x, _ = get_p2(Z_scaled, model, 1,
                  idx_input,
                  bool_mcar=0,
                  training = False,
                  bool_omit_data = bool_omit_data,
                  bool_partial_read = bool_partial_read,)

#import ipdb;ipdb.set_trace()
save_prob_wrap(attackPath,
               p_r_x, 
               idx_mask,
               bool_full = True)

for bool_mcar in [False,True]:
    
    
    mask_all_0 = get_mask_wrap(Z_scaled, model, 
                          blamm_param['seed_model'],
                          idx_input,
                          idx_mask,
                          n_rep = n_rep,
                          bool_mcar = bool_mcar,
                          bool_omit_data = bool_omit_data,
                          bool_partial_read = bool_partial_read,
                          )
    
    mask_all = mask_all_0[:,:,:-1]      #I think because we do not mask label
    mask_all2 = mask_all.reshape(n_rep,-1)
    mask_all_sp = scipy.sparse.csr_matrix(mask_all2)

    with open(os.path.join(attackPath,'mask_mcar_%r.p'%bool_mcar), "wb") as f:
    	pickle.dump(mask_all_sp, f)
        
    #mask_all[0,:,0][bool_omit_data.astype(bool)].mean()
    #p_r_x.numpy()[~bool_omit_data.astype(bool)].mean(0)

#%%
from helper_draw import draw_loss_beta, draw_beta, draw_obj, draw_loss_clsf

#loss_type  = 9
#fig = draw_loss(history, loss_type, bool_bivar = True)
#fig = draw_loss_beta(history)
fig = draw_loss_clsf(history)
fig.savefig(os.path.join(path,'loss_11_28.png'), 
            dpi=200, bbox_inches='tight')

if 0:
    fig = draw_obj(history)
    fig.savefig(os.path.join(path,'obj.png'), 
            dpi=200, bbox_inches='tight')

    fig, ax = plt.subplots()
    
    draw_beta(ax,history,idx_target,idx_omit)
    
    fig.savefig(os.path.join(path,'beta_11_28.png'), 
                dpi=200, bbox_inches='tight')
    
    print('date',path.split('_')[-1])

#%%
#fig, ax = plt.subplots()
#
##ax = axs[0]
#prob = p_r_x.numpy()[:,0]
#
#if family == 'lr':
#    for i in range(2):
#        bool_i = y_train_all==i
#        x_i = X_train_all[bool_i,idx_target]
#        prob_i = prob[bool_i]
#        idx_sort = np.argsort(x_i)
#        
#        ax.plot(x_i[idx_sort],
#                prob_i[idx_sort],'-*',label='y=%i'%i)
#else:
#    x_i = X_train_all[:,idx_target[0]]
#    prob_i = prob
#    idx_sort = np.argsort(x_i)
#    
#    ax.plot(x_i[idx_sort],
#            prob_i[idx_sort],'-*',
##            label='y=%i'%i
#            )
#    
#ax.set_xlabel(r'$X_t$' + ' (%s)'%feat_names[idx_target[0]])
#ax.set_ylabel(r'$P_{R\mid X,y;\phi}(1;x,y)$')
#ax.legend()    
#fig.savefig(os.path.join(path,'p_r_x.png'), 
#            dpi=200, bbox_inches='tight')

#%%