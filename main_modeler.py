# -*- coding: utf-8 -*-
'''
The modelers perspective:
    
https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py

df2:        err_a
df2_2:      err_p
'''
import os
import numpy as np
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
sharedPath = os.path.join(currPath,'results')

import helper_save
from helper_load import get_prob

import helper_glm_main
import helper_glm_wrap
import helper_tex
import helper_blamm_main
import helper_erm_modeler
import helper_mnar

parser = argparse.ArgumentParser()
parser.add_argument('modeler',
                    type=int)
parser.add_argument('grf',
                    type=int)
parser.add_argument('--baseline',
                    type=int,
                    default=0,
                    help='Description for baseline_mnar argument (default: 0)'
                    )
args = parser.parse_args()
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
if not args.baseline:
    blamm_param = helper_blamm_main.get_blamm_param(name, family,
                                                    omit_prop = omit_prop)
else:
    print('using baseline mnar')
#    blamm_param = helper_mnar.get_mnar_params()
    blamm_param = helper_mnar.get_mnar_params(omit_prop)
#%%
Z = np.concatenate([X_train_all,y_train_all[:,np.newaxis]],1)

#%%
print('shp',X_train_all.shape[0]+X_test.shape[0],
      X_train_all.shape[1])

basePath = os.path.join(sharedPath,
#                        'save_nn',
                        'save_nn_repeat',
                        'family_%s'%family,
                        name,
                        )

#import ipdb;ipdb.set_trace()
path = helper_save.get_path2(basePath,
                            [attack_param,blamm_param],
                            )

print('date',path.split('_')[-1])

attackPath = os.path.join(path,'attacker')

with open(os.path.join(attackPath,'mask_mcar_False.p'), "rb") as f:
	mask_all_sp = pickle.load(f)

with open(os.path.join(attackPath,'mask_mcar_True.p'), "rb") as f:
	mask_all_sp_mcar = pickle.load(f)

if args.modeler == 0:
    modeler_l = ['impute','cca']
elif args.modeler == 1:
    modeler_l = ['mice']
elif args.modeler == 2:
    modeler_l = ['linear_lr']
elif args.modeler == 3:
    modeler_l = ['linear_random']
elif args.modeler == 4:
    modeler_l = ['linear']
elif args.modeler == 5:
    modeler_l = ['impute']
elif args.modeler == 6:
    modeler_l = ['cca']
elif args.modeler == 7:
    modeler_l = ['grf']

#%%
if name in ['ate','twins','twins_ns']:
    task = 'ate'
else:
    task = ''
#%%
#'mlp','forest'    
if args.grf == 0:
    estimator_l = ['linear','tnet','tarnet']
elif args.grf == 1:
    estimator_l = ['grf']
elif args.grf == 2:
    estimator_l = ['linear']
elif args.grf == 3:
    estimator_l = ['grf_dr']
elif args.grf == 4:
    estimator_l = ['tf']

for type_modeler in modeler_l:
    
    for type_estimator in estimator_l:
        kwargs_modeler = helper_erm_modeler.get_params(type_estimator,
                                                       blamm_param)        
    
        kwargs_modeler_save, hash_ = helper_save.get_hash(kwargs_modeler)
        modelerPath = os.path.join(path,'modeler_%s'%type_modeler,
                                   'estimator_%s'%type_estimator,
                                   hash_)
        if type_estimator == 'linear':
            figure_path = modelerPath
        else:
            figure_path = None
        
        if not os.path.exists( modelerPath):
        	os.makedirs( modelerPath)
        
        df_temp = pd.Series(kwargs_modeler_save)
        df_temp.to_csv(os.path.join(modelerPath,'param'+'.csv'),
                      header=False
    #                  index=None
                      )
        print('MNAR')
        scores_mnar = helper_erm_modeler.estimate_with_mask(
                                                mask_all_sp,
                                                X_train_all, 
                                                y_train_all,
                                                X_test, y_test,
                                                X_target, y_target,
                                                family,
                                                task,
                                                type_modeler = type_modeler,
                                                feat_names = feat_names,
                                                bool_omit_data = bool_omit_data,
                                                figure_path = figure_path,
                                                figure_prefix = 'mnar',
#                                                type_scaler = blamm_param['type_scaler'],
                                                type_scaler = blamm_param.get('type_scaler'),
                                                type_estimator = type_estimator,
                                                **kwargs_modeler)
             
         
        print('MCAR')
        scores_mcar = helper_erm_modeler.estimate_with_mask(
                                                mask_all_sp_mcar,
                                                X_train_all, 
                                                y_train_all,
                                                X_test, y_test,
                                                X_target, y_target,
                                                family,
                                                task,
                                                type_modeler = type_modeler,
                                                feat_names = feat_names,
                                                bool_omit_data = bool_omit_data,
                                                figure_path = figure_path,
                                                figure_prefix = 'mcar',
#                                                type_scaler = blamm_param['type_scaler'],
                                                type_scaler = blamm_param.get('type_scaler'),
                                                type_estimator = type_estimator,
                                                **kwargs_modeler)
            
        name_metric = helper_glm_wrap.get_metric_name(family)
    
        temp = pd.read_csv(os.path.join(attackPath,'p_r.csv'))
        prob_sum = get_prob(temp)
        prob_mis = (1-prob_sum)*100
        
        df2 = pd.DataFrame({'miss':[np.round(prob_mis,1).astype(str)]})
        df2_2 = pd.DataFrame({'miss':[np.round(prob_mis,1).astype(str)]})
        
        if task == 'ate':
            score_names = ['ate','ate_0','ate_1','auc_0','auc_1']
        
        for i, (scores_i, str_i) in enumerate(zip([scores_mnar,scores_mcar],
                                                   ['mnar','mcar'])):
            
            if family == 'clsf':
                scores_i = scores_i*100
                
            scores_i_pm = helper_save.get_pm(scores_i)
            df_i = pd.DataFrame(scores_i_pm, score_names, 
                                [type_estimator]).T
            df_i.to_csv(os.path.join(modelerPath,'results_%s.csv'%str_i))
            
            df_i_all = pd.DataFrame(scores_i, columns = score_names, 
                                index = [type_estimator]*len(scores_i))
            df_i_all.to_csv(os.path.join(modelerPath,'results_all_%s.csv'%str_i))
        

print('date',path.split('_')[-1])

#%%
fig, axs = plt.subplots(1,3,
#                        sharey='all',
                        sharex='all')

#ax = axs[1]

if family == 'lr':
    ax = axs[0]
    for i in range(2):
        bool_i = y_train_all==i
        x_i = X_train_all[bool_i,idx_target]
        
        ax.hist(x_i,label='y=%i'%i)
        ax.legend()    
else:
    x = X_train_all[:,idx_target]
    axs[0].hist(x)
    for i in range(2):
        
        if i == 0:
            mask_in = mask_all_sp
        else:
            mask_in = mask_all_sp_mcar
            
        mask_all = np.array(mask_in.todense()).reshape(-1,*X_train_all.shape)
        mask_i = mask_all[0]

        x_mask = x[~mask_i[:,idx_target]]
        axs[1+i].hist(x_mask)
        
#ax.set_xlabel(r'$X_t$' + ' (%s)'%feat_names[idx_target[0]])
#ax.set_ylabel(r'$P_{R\mid X,y;\phi}(1;x,y)$')
fig.set_size_inches( w = 15,h = 5)	
fig.savefig(os.path.join(path,'x_hist.png'), 
            dpi=200, bbox_inches='tight')

#%%
from helper_prob.metrics import getCorr_XY

corr_xy = getCorr_XY(X_train_all,y_train_all[:,np.newaxis])[:,0]
df1 = pd.DataFrame({'Pearson(X_j,y)':corr_xy},
                    index=feat_names)

df1 = df1.round(2)
df1.to_csv(os.path.join(path,'corr_xy.csv'))
helper_tex.save_tex(path, 'corr_xy', df1,
                    index=True)

corr = getCorr_XY(X_train_all,X_train_all)

df2 = pd.DataFrame(corr,
                   columns=feat_names,
                   index=feat_names)

df2=df2.round(2)
df2.to_csv(os.path.join(path,'corr_x.csv'))
helper_tex.save_tex(path, 'corr_x', df2,
                    index=True)