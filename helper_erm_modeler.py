# -*- coding: utf-8 -*-
import numpy as np
import copy

from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from helper_glm_wrap import get_impute
import matplotlib.pyplot as plt

import os

import helper_draw_nn
from helper_erm_nn import _get_solver
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from helper_scale import get_scaled_sklearn
    
import inspect
import shlex
import subprocess

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
sharedPath = os.path.join(currPath, 'results')

try:
    import helper_grf
except:
    print('rpy2 not installed')

def estimate_with_mask(mask_all_sp,
                       X_train_all, 
                       y_train_all,
                       X_test, y_test,
                       X_target, y_target,
                       family,
                       task,
                       type_estimator,
                       type_modeler = 'impute',
#                       bool_bias = False,
                       feat_names = None,
                       kwargs_data_val = None,
                       bool_omit_data = None,
                       **kwargs_modeler):
    
    '''
    Modeler's side
    '''
                
    mask_all = np.array(mask_all_sp.todense()).reshape(-1,*X_train_all.shape)

#    import ipdb;ipdb.set_trace()    
    print('features_with missingness',
          [feat_names[idx_i] for idx_i in np.where(np.any(mask_all,(0,1)))[0]])
    
    if bool_omit_data is not None:      #Sanity check
#        import ipdb;ipdb.set_trace()
        if mask_all[:,~bool_omit_data.astype(bool)].any():
            raise ValueError
    
    if kwargs_data_val is None:
        scores = _estimate_masked(X_train_all, y_train_all, 
                                  X_test, y_test,
                                  X_target, y_target,
                                   mask_all,
                                   family = family,
                                   task = task,
                                   type_modeler = type_modeler,
#                                                   bool_bias = bool_bias,
                                   feat_names = feat_names,
                                   type_estimator = type_estimator,
                                   **kwargs_modeler)
    else:
        pass
        
    return scores
            
def _estimate_masked(X_train_all, y_train_all_0, 
                     X_test_0, y_test,
                     X_target_0, y_target,
                     mask_all,
                     family,
                     type_modeler,
                     task,
                     type_estimator,
                     bool_shuffle = False,
#                     bool_bias = False,
                     feat_names = None,
                     figure_path = None,
                     figure_prefix = '',                     
                     type_scaler = None,
#                     n_mask_0 = -1,
                     **kwargs_modeler
                     ):
    
    '''
    For each mask matrix, runs the modeler
    
    In:
        mask_all:       #
    '''
#    n_mask = len(mask_all)    
#    bool_draw_clsf = (family == 'clsf') and len(feat_names)
    if (figure_path is not None) and (task != 'ate'):
#        fig, ax = plt.subplots()
        model_complete = _estimate(X_train_all, y_train_all_0,
                               family,
#                               verbose = i == 0,
                               feat_names = feat_names,
                               **kwargs_modeler)
        
    n_mask = min(len(mask_all),5)
#    n_mask = len(mask_all)
    
    bool_d = [np.array_equal(np.unique(X_j),[0,1]) for X_j in X_train_all.T]
    idx_d = np.where(np.array(bool_d))[0]
    
    if len(idx_d)>0:
        if feat_names is None:
            print('idx_d', idx_d)
        else:
            print('idx_d', [feat_names[idx_i] for idx_i in idx_d])
    
    temp = []
    for i in range(n_mask):
        
        mask = mask_all[i]
        
        if bool_shuffle:
            idx = np.arange(len(mask))
            np.random.seed(i)
            np.random.shuffle(idx)
            mask = mask[idx]
            
        X_miss_0 = copy.deepcopy(X_train_all)
        X_miss_0[mask] = np.nan
        
        #Drops the completely missing rows
        bool_all_miss = np.isnan(X_miss_0).all(1)
        X_miss = X_miss_0[~bool_all_miss]
        y_train_all_1 = y_train_all_0[~bool_all_miss]
        
        if bool_all_miss.any():
            print('%.2f per. rows are dropped'%
                  (bool_all_miss.sum()/len(X_train_all)*100))
#        if i ==0:
#            print('mean',np.round(np.nanmean(X_miss,0),2))
        
        #%%
        if type_modeler != 'grf':
            X_hat_0, y_train_all = get_impute(type_modeler, 
                                            X_miss, y_train_all_1,
                                            seed = i,
                                            idx_d = idx_d)
        else:
            X_hat_0 = X_miss
            y_train_all = y_train_all_1
        
#        import ipdb;ipdb.set_trace()
        
        if type_scaler is None:
            X_hat = X_hat_0
            X_test = X_test_0
        else:        
            X_hat, X_test, X_target = get_scaled_sklearn(X_hat_0, X_test_0, 
                                                         X_target_0, 
                                                         type_scaler)
                
#        import ipdb;ipdb.set_trace()
        if (task != 'ate'):
            if type_estimator == 'mlp':
                model = _estimate(X_hat, y_train_all,
                                family,
                                verbose = i == 0,
                                feat_names = feat_names,
                                **kwargs_modeler)
                
                if family == 'clsf':
                    acc, auc = _evaluate(model, 
                                         X_test, y_test,
                                         family)
                    
                    prob = _get_prob(model, 
                                     X_target, y_target,
                                     family)
                    scores = [acc, auc, prob]
                elif family == 'reg':
#                    ate = get_ate(model)                
                    scores = [ate]
            else:
                if (family == 'clsf'):
                    acc, auc, prob = _estimate_and_score_logistic(X_hat, y_train_all,
                                                            X_test, y_test,
                                                            X_target,
                                                            family)
                    scores = [acc, auc, prob]
        else:               
            if type_estimator not in ['tnet','tarnet','grf','grf_dr']:
                ate, ate_0, ate_1, \
                auc_0, auc_1 = _estimate_and_score_ate(X_hat, y_train_all,
                                              X_test, y_test,
                                              family,
                                              type_estimator)
            elif type_estimator in ['tnet','tarnet']:
                ate, ate_0, ate_1, \
                auc_0, auc_1 = estimate_catenets_wrapped(X_hat, y_train_all,
                                                         X_test, y_test,
                                                         family,
                                                         type_estimator)
            elif type_estimator == 'grf':
                ate, ate_0, ate_1, \
                auc_0, auc_1 = helper_grf._estimate_and_score_ate(X_hat, y_train_all,
                                                                  X_test, y_test,
                                                                  family,
                                                                  type_estimator)
            elif type_estimator == 'grf_dr':
                print('using DR')
                ate, ate_0, ate_1, \
                auc_0, auc_1 = helper_grf._estimate_and_score_ate_dr(X_hat, y_train_all,
                                                                     X_test, y_test,
                                                                     family,
                                                                     type_estimator,
                                                                     **kwargs_modeler)
                            
            scores = [ate, ate_0, ate_1, auc_0, auc_1]                
        
        temp.append(scores)
        
        if figure_path is not None:
            if (family == 'clsf') and task != 'ate':
                fig = draw_figure_clsfy()
            else:
#                import ipdb;ipdb.set_trace()
                fig = draw_figure_ate(X_train_all, y_train_all_0,
                                      X_hat_0, y_train_all,
                                      X_miss_0, X_miss,
                                      type_modeler)

            os.makedirs(os.path.join(figure_path,'figures'), exist_ok=True)
            
            if i < 5:
                fig.savefig(os.path.join(figure_path,
                                         'figures',
                                         '%s_est_boundary_%i.png'%(figure_prefix,i)),
        	            dpi=200, bbox_inches='tight')
    
    scores = np.stack(temp,0)
    return scores

def _estimate(X_0, y,
              family,
#              rtol=1e-2
              solver_layers, 
              solver_epochs,
              solver_lr,
              solver_arch = 'mlp',
              solver_seed = 42,
              solver_lambda = 0,
              solver_optimizer = 'adam',
              verbose = False,
              feat_names = None,
              ):
    
    p = X_0.shape[1]
    
    if 0:
        tf.random.set_seed(seed)
            
        model = Sequential()
        model.add(keras.Input(shape=(p,)))
            
    #    import ipdb;ipdb.set_trace()
        for unit in nn_layers:
            model.add(layers.Dense(unit, activation='sigmoid'))
        
        model.add(layers.Dense(1,activation=None))
    
    model = _get_solver(p, solver_layers, 
                        solver_arch,
                        seed_solver = solver_seed,
                        solver_lambda = solver_lambda,
                        bool_tilde = False)
        
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    
    if solver_optimizer == 'adam':
        model.compile(
                      optimizer=keras.optimizers.Adam(learning_rate=solver_lr),
                      loss=loss_fn,
                      run_eagerly = 1,
                      )
    
    model.fit(x=X_0,
              y=y,
              batch_size=len(X_0),
              epochs=solver_epochs,
              verbose= verbose)
    
    return model

def _evaluate(model, 
              X, y,
              family):
    
    if family == 'clsf':
        y_score = model(X)
        y_pred = y_score>0
        
        acc = accuracy_score(y, y_pred.numpy())
        auc = roc_auc_score(y, y_score.numpy())                
        
        return acc, auc
    
def _get_prob(model, 
              X, y,
              family):
    
    if family == 'clsf':
        y_score = model(X)
        
        prob = tf.math.sigmoid(y_score).numpy().mean(0)
        
        return prob
    
def _estimate_and_score_logistic(X_train, y_train,
                           X_test, y_test,
                           X_target,
                           family):
    
    if family == 'clsf':
        model = LogisticRegression(solver='saga', 
                           random_state=42, 
                           max_iter=1000)
        model.fit(X_train, y_train)    
        print("Model training complete.")
    
#        import ipdb;ipdb.set_trace()
        y_score = model.predict_proba(X_test)[:,1]
        y_pred = y_score>0.5
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)

        prob = model.predict_proba(X_target)[:,1].mean(0)
        
        return acc, auc, prob
    
def _estimate_and_score_ate(X_train, y_train,
                           X_test, y_test,
                           family,
                           type_estimator):
    
    if family == 'reg':
        model = LinearRegression()
    elif family == 'clsf':
        print('using %s'%type_estimator)
        if type_estimator == 'linear':
            model = LogisticRegression(solver = 'lbfgs',
#                                        solver='saga', 
                                       max_iter= 600,
                                       penalty = 'none')
        elif type_estimator == 'mlp':
            model = MLPClassifier(max_iter=400)
            print('using MLP')
        elif type_estimator == 'forest':
            model = RandomForestClassifier()
            print('using forest')
        
    model.fit(X_train, y_train)
    print("Model training complete.")

#    X = X_train[:,:-1]
#    X = X_test[:,:-1]    
#    print('using test')
    ate, ate_0, ate_1 = _score_ate(X_test, model, family)
    auc_0, auc_1 = _score_auc(X_test, y_test, model, family)
#    X_0 =np.concatenate([X,np.zeros([len(X),1])],1)
#    X_1 = np.concatenate([X,np.ones([len(X),1])],1)
#    
#    if family == 'reg':        
#        est_0 = model.predict(X_0)
#        est_1 = model.predict(X_1)
#    elif family == 'clsf':        
#        est_0 = model.predict_proba(X_0)[:,1]
#        est_1 = model.predict_proba(X_1)[:,1]
#    
#    ate = np.mean(est_1-est_0)
#    ate_1 = np.mean(est_1)
#    ate_0 = np.mean(est_0)
    
    return ate, ate_0, ate_1, auc_0, auc_1

def _score_ate(X_in, model, family):
    
    #    X = X_train[:,:-1]
    X = X_in[:,:-1]
#    print('using test')
    X_0 =np.concatenate([X,np.zeros([len(X),1])],1)
    X_1 = np.concatenate([X,np.ones([len(X),1])],1)
    
    if family == 'reg':        
        est_0 = model.predict(X_0)
        est_1 = model.predict(X_1)
    elif family == 'clsf':        
        est_0 = model.predict_proba(X_0)[:,1]
        est_1 = model.predict_proba(X_1)[:,1]
    
    ate = np.mean(est_1-est_0)
    ate_0 = np.mean(est_0)
    ate_1 = np.mean(est_1)
    
    return ate, ate_0, ate_1

def _score_auc(X_in, y, model, family):
    
    '''
    In:
        y:      N,2             Contains both outcomes
    '''
    
    #    X = X_train[:,:-1]
    X = X_in[:,:-1]
#    print('using test')
    X_0 = np.concatenate([X,np.zeros([len(X),1])],1)
    X_1 = np.concatenate([X,np.ones([len(X),1])],1)
    
    if family == 'reg':
        raise ValueError
    elif family == 'clsf':
        est_0 = model.predict_proba(X_0)[:,1]
        est_1 = model.predict_proba(X_1)[:,1]
        
    auc_0 = roc_auc_score(y[:,0], est_0)
    auc_1 = roc_auc_score(y[:,1], est_1)
    
    return auc_0, auc_1

def draw_figure_clsfy(X_train_all, 
                     y_train_all_0, 
                     X_target, 
                     y_target,
                     X_hat,
                     y_train_all,
                     X_miss_0,
                     X_miss,
                     model_complete,
                     model,
                     type_modeler):
    
    fig, axs = plt.subplots(1,3)
    x_min, x_max, \
    y_min, y_max = helper_draw_nn.draw_decision_boundary(X_train_all, 
                                                         y_train_all_0, 
                                                         X_target, 
                                                         y_target,
                                                         model_complete, 
                                                         axs[0])
    
#            bool_miss = np.isnan(X_miss_0).any(1)
    bool_miss = np.isnan(X_miss_0)
#            import ipdb;ipdb.set_trace()
    helper_draw_nn.draw_decision_boundary(X_train_all, 
                                          y_train_all_0, 
                                          X_target, 
                                          y_target,
                                          model_complete, axs[1],
                                          x_min, x_max,
                                          y_min, y_max,
                                          bool_miss=bool_miss)
    
    if type_modeler != 'cca':
        bool_miss = np.isnan(X_miss)
    else:
        bool_miss = None
    helper_draw_nn.draw_decision_boundary(X_hat, 
                                          y_train_all, 
                                          X_target, 
                                          y_target,
                                          model, axs[2],
                                          x_min, x_max,
                                          y_min, y_max,
                                          bool_miss=bool_miss)
    
    
    axs[0].set_title("Complete data")
    axs[1].set_title("Partially observed")
    axs[2].set_title("After remediation")
    
    #import seaborn as sns
    fig.set_size_inches( w = 15,h = 5)
    return fig

def draw_figure_ate(X_0, y_0, 
                    X_hat_0, y_hat_0,
                    X_miss_0, X_miss_hat,
                    type_modeler):
    skip = 10
    
    bool_miss_0 = np.isnan(X_miss_0[::skip])
    idx_target = np.where(bool_miss_0.any(0))[0][0]
    bool_miss = bool_miss_0[:,idx_target]
    
    X = X_0[::skip,idx_target]
    W = X_0[::skip,-1]
    y = y_0[::skip]
    
    X_hat = X_hat_0[::skip,idx_target]
    W_hat = X_hat_0[::skip,-1]
    if type_modeler != 'cca':
        bool_miss_hat = np.isnan(X_miss_hat[::skip,idx_target])
    else:
        bool_miss_hat = np.zeros_like(X_hat,bool)
    y_hat = y_hat_0[::skip]
    
    fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
    
    for i in range(2):        
        for ii in range(2):
            
            if ii == 0:
                bool_sub = W == i
                X_sub = X[bool_sub]
                y_sub = y[bool_sub]
                bool_miss_sub = bool_miss[bool_sub]
                title_str = 'Partially observed'
            else:
                bool_sub = W_hat == i
                X_sub = X_hat[bool_sub]
                y_sub = y_hat[bool_sub]
                bool_miss_sub = bool_miss_hat[bool_sub]
                title_str = 'After remediation'
                
            axs[i,ii].scatter(X_sub[~bool_miss_sub], 
                               y_sub[~bool_miss_sub], 
                               alpha=0.5,
                               c='tab:blue',
                               label='complete')
            axs[i,ii].scatter(X_sub[bool_miss_sub], 
                               y_sub[bool_miss_sub], 
                               alpha=0.5,marker='x',
                               c='tab:orange',
                               label='partial')
            
            slope, intercept = np.polyfit(X_sub, y_sub, 1)
            x_line = np.linspace(min(X_sub), max(X_sub), 100)
            y_line = slope * x_line + intercept            
            # Plot the regression line
            axs[i,ii].plot(x_line, y_line, color='tab:green', 
                           linewidth=2)
            axs[i,ii].grid(True, which='both', linestyle='--', linewidth=0.5)


            axs[i,ii].set_title('%s | W=%i'%(title_str,i))
            axs[i,ii].legend()
            axs[i,ii].set_xlabel('X (Covariate)')
            axs[i,ii].set_ylabel('Y (Outcome)')
            axs[i,ii].xaxis.set_tick_params(which='both', labelbottom=True, bottom=True)
            axs[i,ii].yaxis.set_tick_params(which='both', labelleft=True, left=True)        
        
    



    fig.set_size_inches( w = 20, h = 10)
    
    return fig

def estimate_catenets_wrapped(X_train_0, y_train,
                              X_test, y_test,
                              family,
                              type_estimator):
    
#    str_arg = '%i %f'%(seed, w_threshold)
#    str_arg = '%i %f %i'%(seed, w_threshold, bool_linear)
    str_arg = '%s %s'%(family, type_estimator)
    script_name = 'call_catenets'
    
    scores = get_script_wrapped(X_train_0, y_train,
                                X_test, y_test,
                                str_arg,
                              script_name,
                              ['scores'],
                              )

#    ate, ate_0, ate_1, \
#    auc_0, auc_1 = scores[0]
#    import ipdb;ipdb.set_trace()
    
#    W, cov_est = load#['W'],load['cov_est']
    
    return scores[0]

def get_script_wrapped(X_train_0, y_train,
                       X_test, y_test,
                       str_arg,
                       script_name,
                       key_l):
    '''
    
    '''
#    sharedPath = shared
#    miniconda = os.path.join(os.environ['HOME'],
#                             'miniconda3/envs/catenets/bin/python'
#                             )    
    miniconda = os.path.join(os.environ['CATEPATH'])    
    
    script_path = os.path.join(currPath,script_name+'.py')
    savePath = os.path.join(sharedPath,'temp_%s_%s'%(script_name,str(os.getpid())))
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    for arr, name in zip([X_train_0, y_train, X_test, y_test],
                         ['X_train','y_train','X_test','y_test']):
        file_x = os.path.join(savePath,'%s.npy'%name)
        if os.path.exists(file_x):
            os.remove(file_x)
            
        with open(file_x, 'wb') as f:
            np.save(f, arr)
    
    file_z = os.path.join(savePath,'temp.npyz')
    if os.path.exists(file_z):
        os.remove(file_z)

    command = "%s %s %s %s"%(miniconda,
                             script_path,
                             savePath,
                             str_arg)
                                                     
    print(command)
#    import ipdb;ipdb.set_trace()
    args = shlex.split(command)
    my_subprocess = subprocess.Popen(args)
    my_subprocess.wait()
#    os.system(command)
    
    
    with open(file_z, 'rb') as f:
        load_0 = np.load(f)
        load = [load_0[key] for key in key_l]
    
    os.remove(file_z)
    os.remove(file_x)
    
    return load

def get_params(type_estimator, blamm_param):
    
    if type_estimator == 'tf':
        kwargs_modeler = blamm_param['kwargs_solver']
        
        #%%
        if 'solver_warm_start' in kwargs_modeler.keys():
            kwargs_modeler.pop('solver_warm_start')
        
        if 'solver_arch' in kwargs_modeler.keys():
            if kwargs_modeler['solver_arch'] == 'rnn_p2012_all':
                kwargs_modeler['solver_epochs'] = 20
            #    kwargs_modeler['solver_lr'] = 2e-3
            else:
                kwargs_modeler['solver_optimizer'] = 'adam'
        else:
        #    kwargs_modeler['solver_layers'] = [10]
            kwargs_modeler['solver_epochs'] = 300
            kwargs_modeler['solver_optimizer'] = 'adam'
            if 'solver_optimizer_kwargs' in kwargs_modeler.keys():
                kwargs_modeler.pop('solver_optimizer_kwargs')
                
        if ('mnist' in name) and (len(kwargs_modeler['solver_layers']) == 0):
            kwargs_modeler['bool_lr'] = True
        
        if 0:
            if kwargs_modeler['solver_layers'] == 0:
                kwargs_modeler['type_predictor'] = 'linear'
            else:
                kwargs_modeler['type_predictor'] = 'mlp'
    else:
        kwargs_modeler = {}

#    if 'type_scaler' in blamm_param.keys():
#        kwargs_modeler['type_scaler'] = blamm_param['type_scaler']    
        
    return kwargs_modeler
    