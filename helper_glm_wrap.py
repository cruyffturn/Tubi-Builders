# -*- coding: utf-8 -*-
import os
import numpy as np
import copy
import pickle
from hashlib import sha1
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except:
    print('tensorflow not installed')
    
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

import statsmodels.api as sm

import helper_tf_model_glm
from helper_load import get_mask, get_p2
import helper_tf_model_irls
import helper_fill

import helper_data_val

def legacy_train(X_train_all,
          y_train_all,
          idx_input,
          idx_mask,
          beta_a,
          seed_model,
          lr,
          epochs,
          reg_lmbda,
          name_optimizer,
          loss_type,
          bool_scale_all
          ):
    
        #scaler = StandardScaler()
    scale_mu = np.mean(X_train_all,0)
    scale_std = np.std(X_train_all,0)
    
    if not bool_scale_all:
        mu_y = 0
        std_y = 1
    else:
        mu_y = np.mean(y_train_all,0)
        std_y = np.std(y_train_all,0)
        print('scaling all')
        
    scale_mu = np.append(scale_mu,[mu_y])
    scale_std = np.append(scale_std,[std_y])
    #%%
    
    def scheduler(epoch, lr_in):
        
        if epoch < 10:
            return lr/100
        else:
            return lr        
    
    tf.random.set_seed(seed_model)
    
    model_cfg = 2
        
    #idx_adv = range(p)
        
    model = helper_tf_model_glm.get_model(model_cfg, 
                                         idx_input,
                                         idx_mask
                                         )
    
    model._set_param(idx_input,
                     idx_mask,
                     beta_a = beta_a,
                     scale_mu=scale_mu,
                     scale_std=scale_std,
                     reg_lmbda = reg_lmbda,
                     loss_type = loss_type)
    
    if name_optimizer == 'adam':
        optim = keras.optimizers.Adam(learning_rate=lr)
        
    elif name_optimizer == 'adagrad':
        optim = keras.optimizers.Adagrad(learning_rate=lr)
                
    model.compile(
    #              optimizer=keras.optimizers.Adam(lr),
                  optimizer=optim,
                  run_eagerly = 1,
                  )
            
    
    min_delta = 1e-4
    callbacks = [
         tf.keras.callbacks.LearningRateScheduler(scheduler)]
    
    
    Z = np.concatenate([X_train_all,y_train_all[:,np.newaxis]],1)
    
    history = model.fit(Z, 
                        epochs=epochs,
                        batch_size=len(Z),
                        callbacks=callbacks
                        )
    
    return model, history, scale_mu, scale_std

def train_irls(X_train_all,
              y_train_all,
              idx_input,
              idx_mask,
              beta_a,
              c_a,
              seed_model,
              lr,
              epochs,
              name_optimizer,
              bool_scale_all,
              bool_bias,
              model_cfg = 2,
              bool_ig = False,
              type_attack = 'glm',
              bool_omit_data = None,
              **kwargs,
              ):
    
        '''        
        Trains the NN For attacking the IRLS algorithm
        '''
        
            #scaler = StandardScaler()
        scale_mu = np.mean(X_train_all,0)
        scale_std = np.std(X_train_all,0)
        
        if not bool_scale_all:
            mu_y = 0
            std_y = 1
        else:
            mu_y = np.mean(y_train_all,0)
            std_y = np.std(y_train_all,0)
            print('scaling all')
            
        scale_mu = np.append(scale_mu,[mu_y])
        scale_std = np.append(scale_std,[std_y])
        #%%
        
        def scheduler(epoch, lr_in):
            
            if epoch < 10:
                return lr/100
#            elif epoch > 600:
#                return lr/100
            else:
                return lr        
        
        tf.random.set_seed(seed_model)        
            
        #idx_adv = range(p)
        p = X_train_all.shape[1]
        
        if type_attack == 'glm':
            model = helper_tf_model_irls.get_model(model_cfg, 
                                         idx_input,
                                         idx_mask,
                                         p,
                                         bool_ig = bool_ig,
                                         bool_omit_data = bool_omit_data
                                         )                   
            
            model._set_param(idx_input,
                             idx_mask,
                             beta_a,
                             c_a,
                             scale_mu=scale_mu,
                             scale_std=scale_std,
                             bool_bias = bool_bias,
                             epochs = epochs,
                             **kwargs)
            
        elif type_attack == 'scm':
            pass
            
                
#        import ipdb;ipdb.set_trace()
        if name_optimizer == 'adam':
            optim = keras.optimizers.Adam(learning_rate=lr)
            
        elif name_optimizer == 'adagrad':
            optim = keras.optimizers.Adagrad(learning_rate=lr)
                    
        model.compile(
        #              optimizer=keras.optimizers.Adam(lr),
                      optimizer=optim,
                      run_eagerly = 1,
                      )
                
        
        min_delta = 1e-4
        callbacks = [
             tf.keras.callbacks.LearningRateScheduler(scheduler)]
        
        
        Z = np.concatenate([X_train_all,y_train_all[:,np.newaxis]],1)
        
        #batch_size has to be all samples for bool_omit_data to work
        history = model.fit(Z, 
                            epochs=epochs,
                            batch_size=len(Z),
                            callbacks=callbacks
                            )
        
        return model, history, scale_mu, scale_std

def estimate(Z, X_train_all, 
             y_train_all,
             scale_mu, scale_std,
             model, seed_model,
             idx_input, idx_mask,
             family,
             type_modeler = 'impute',
             bool_bias = False,
             n_rep = 20):
    
    '''
    Legacy: modeler's side
    '''
    Z_scaled = (Z - scale_mu)/scale_std
    
    #Sets the seed
    np.random.seed(42)
#    print('tf seed', seed_model)
    tf.random.set_seed(seed_model)
    
    p_r_x, _ = get_p2(Z_scaled, model, 1,
                       idx_input,
                       bool_mcar=0)

    #
    mask_all_0 = get_mask(Z_scaled, model, 
                          seed_model,
                          idx_input,
                          idx_mask,
                          n_rep = n_rep,                    
                          )
    
    mask_all = mask_all_0[:,:,:-1]      #I think because we do not mask label
    
    beta_est_all, c_est_all = _estimate_masked(X_train_all, y_train_all, 
                                               mask_all,
                                               family = family,
                                               type_modeler = type_modeler,
                                               bool_bias = bool_bias)
    
    beta_est_all_mcar, c_est_all_mcar = _estimate_masked(X_train_all, y_train_all, 
                                                         mask_all,
                                                         bool_shuffle = 1,
                                                         family = family,
                                                         type_modeler = type_modeler,
                                                         bool_bias = bool_bias)
    
    
    #%%
        
    return beta_est_all, c_est_all, \
           beta_est_all_mcar, c_est_all_mcar, \
           p_r_x

def _estimate_masked(X_train_all, y_train_all_0, 
              mask_all,
              family,
              type_modeler,
              bool_shuffle = False,
              bool_bias = False,
              feat_names = None):
    
    '''
    For each mask matrix, runs the modeler
    
    In:
        mask_all:       #
    '''
    n_mask = len(mask_all)
    
    beta_est_all = np.zeros([n_mask,mask_all.shape[-1]])
    c_est_all = np.zeros([n_mask])
    
    pvalues_all = np.zeros([n_mask,mask_all.shape[-1]+1])
    tvalues_all = np.zeros([n_mask,mask_all.shape[-1]+1])
    
    bool_d = [np.array_equal(np.unique(X_j),[0,1]) for X_j in X_train_all.T]
    idx_d = np.where(np.array(bool_d))[0]
    
    if len(idx_d)>0:
        if feat_names is None:
            print('idx_d', idx_d)
        else:
            print('idx_d', [feat_names[idx_i] for idx_i in idx_d])
    
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
        X_hat, y_train_all = get_impute(type_modeler, 
                                        X_miss, y_train_all_1,
                                        seed = i,
                                        idx_d = idx_d)
        
        beta_est, c_est,\
        pvalues, tvalues = _estimate(X_hat, y_train_all,
                                    family,
                                    verbose = i == 0,
                                    bool_bias = bool_bias,
                                    return_pval = 1,
                                    feat_names = feat_names)
        
        temp_beta, temp_c = _estimate(X_hat, y_train_all,
                                      family,
                                      verbose = False,
                                      bool_bias = bool_bias)
        
        if not np.allclose(beta_est,temp_beta) or \
            not np.allclose([c_est], [temp_c]):            
#            import ipdb;ipdb.set_trace()
            raise ValueError
#        else:
#            print('check success')
            
#        from helper_tf_glm import get_irls_enum_det
#        print('using enum')
#        beta_est, c_est = get_irls_enum_det(X_hat, y_train_all, 
#                                            family,
#                                            bool_intercept = True,
#                                            n_steps = 1,
#                                            bool_while = True
#                                            )
        
#        import ipdb;ipdb.set_trace()

#        print('beta',np.round(beta_est,2))
        
        beta_est_all[i] = beta_est
        c_est_all[i] = c_est
        pvalues_all[i] = pvalues
        tvalues_all[i] = tvalues
        
    return beta_est_all, c_est_all, \
            pvalues_all, tvalues_all

def _estimate(X_0, y,
              family,
              verbose = False,
              bool_bias = False,
              return_pval = False,
              feat_names = None,
#              rtol=1e-2
              rtol=5e-1
              ):
    
    '''
    Given X,y estimates the coefficients
    '''
#    import ipdb;ipdb.set_trace()
    if not verbose:
        _kwargs = {'disp':0}
    else:
        _kwargs = {}
    
    if feat_names is None:
        X = X_0
    else:
        X = pd.DataFrame(X_0, columns = feat_names)
    
    if bool_bias:
        X_in = sm.add_constant(X)
    else:
        X_in = X
            
    if family == 'normal':
#        beta_est = (np.linalg.inv(X.T @ X) @ X.T).dot(y)
        
        model = sm.OLS(y, X_in)
        results = model.fit()        
            
    elif family == 'lr':        
            
        model = sm.Logit(y, X_in)
        
        init = np.zeros(X_in.shape[1])#model2.coef_.tolist()
        results = model.fit(start_params=init,
                            **_kwargs,
#                            maxiter=100
        #                    method='lbfgs',
#                            maxiter=1
                            )
#        print('wraning maxiter=1')
    if verbose: print(results.summary())
    
#    import ipdb;ipdb.set_trace()
    pvalues = results.pvalues
    tvalues = results.tvalues
    beta_est_0 = results.params
    
    if feat_names is not None:
        pvalues = pvalues.to_numpy()
        tvalues = tvalues.to_numpy()
        beta_est_0 = beta_est_0.to_numpy()
    
    if not bool_bias:
        beta_est = beta_est_0
        pvalues = np.hstack([pvalues,-1])
        tvalues = np.hstack([tvalues,-1])
    else:
        c = beta_est_0[0]
        beta_est = beta_est_0[1:]
        pvalues = np.hstack([pvalues[1:],pvalues[0]])
        tvalues = np.hstack([tvalues[1:],tvalues[0]])
    
    if not bool_bias:
        c = 0
    
    if family == 'normal':
        model2 = LinearRegression(fit_intercept=bool_bias)
        model2.fit(X_0, y)
        beta_est2 = model2.coef_        
        
        if bool_bias:
            c2 = model2.intercept_
        else:
            c2 = 0
        
        print('comparing')
        if not np.allclose(beta_est2,beta_est,rtol=rtol): 
            import ipdb;ipdb.set_trace()
            raise ValueError
        if not np.allclose([c2],[c]): 
            import ipdb;ipdb.set_trace()
            raise ValueError
            
            
    if not return_pval:    
        return beta_est, c
    else:        
        return beta_est, c, pvalues, tvalues

def get_impute(type_modeler,
               X_miss, y_0,
               seed = 42,
               idx_d = None):
    
    '''
    Inputs the missing data in X_miss
    '''
    
    if type_modeler == 'impute':
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_hat = imp_mean.fit_transform(X_miss).astype(np.float32)
        
    elif type_modeler == 'cca':
        bool_any_miss = np.isnan(X_miss).any(1)
        X_hat = X_miss[~bool_any_miss]
        
    elif type_modeler == 'mice':
        X_hat = helper_fill.get_mice(X_miss, y_0,
                                     seed = seed,
                                     idx_d = idx_d)
    
    elif type_modeler == 'linear':
        X_hat = helper_fill.get_linear(X_miss, y_0,
                                       seed = seed)
    
    elif type_modeler == 'linear_lr':
        X_hat = helper_fill.get_linear(X_miss, y_0,
                                       family = 'lr',
                                       seed = seed)
        
    elif type_modeler == 'linear_random':
        X_hat = helper_fill.get_linear(X_miss, y_0,
                                       seed = seed,
                                       bool_random = True)
        
    
    if type_modeler == 'cca':
        y = y_0[~bool_any_miss]
    else:
        y = y_0
        
    return X_hat, y
            
def get_metric(X, y, 
             beta_all, 
             c_all,
             family,
             y_train):
    
    '''
    In:
        X:  N,p
        y:  N,
        beta_all:   K,p
        c_all:      K,
        err:        K,
        y_train:    N_train         #Used for nMSE
        
    Inter:
        eta:    N,
    '''        
    err_all = np.zeros(len(beta_all))
    
    for k, (beta,c) in enumerate(zip(beta_all,c_all)):
        
        eta = X.dot(beta) + c
        
        if family == 'normal':
            err_0 = np.mean((eta - y)**2)
            err_1 = np.mean((y_train.mean()-y)**2)
            
            metric = err_0/err_1
            
        elif family == 'lr':
            
            y_est = eta > 0
            metric = (y_est == y).mean()
                    
        
        err_all[k] = metric
        
    return err_all

def get_metric_name(family):
    
    if family == 'normal':
        return 'NMSE'
    
    elif family == 'lr':
        return 'Accuracy'

def save_model(modelPath, 
               X_train, y_train,
               model, history,
               scale_mu, scale_std):

    if not os.path.exists( modelPath):
        os.makedirs( modelPath)

    with open(os.path.join(modelPath,'history.p'), "wb") as f:
        pickle.dump([history.history], f)
        
    with open(os.path.join(modelPath,'scale.p'), "wb") as f:
        pickle.dump([scale_mu, scale_std], f)
        
    with open(os.path.join(modelPath,'optim.p'), "wb") as f:
        pickle.dump([model.optimizer.get_weights(),
                     model.optimizer.get_config()], f)
    
    hash_x = sha1(X_train).hexdigest()
    hash_y = sha1(y_train).hexdigest()
    
    with open(os.path.join(modelPath,'%s_%s'%(hash_x,hash_y)), "wb") as f:
        pickle.dump([hash_x,hash_y], f)
        
    #Saves the model
    model.save(os.path.join(modelPath,'model'))
    
#    Saves the data
#    with open(os.path.join(simPath,'data.p'), "wb") as f:
#        pickle.dump(load, f)    
    

def load_model(modelPath, X_train, y_train):
    
    hash_x = sha1(X_train).hexdigest()
    hash_y = sha1(y_train).hexdigest()
    
    if not os.path.exists(os.path.join(modelPath,'%s_%s'%(hash_x,hash_y))):
        raise TypeError
        
    with open(os.path.join(modelPath,'scale.p'), "rb") as f:
        scale_mu, scale_std = pickle.load(f)
        
    custom_objects = dict(Custom=helper_tf_model_irls.Custom)

    savePath2 = os.path.join(modelPath,'model')    
        
    model = tf.keras.models.load_model(savePath2, 
                                       custom_objects=custom_objects)

    return model, scale_mu, scale_std
#
#def get_setup_wrap():
#    
#    (X, mu, S, 
#                sigma_sq, W, 
#                idx_adv_train,
#                idx_mask,
#                mu_a, S_a, W_a)
    
def estimate_with_mask(mask_all_sp,
                       X_train_all, 
                       y_train_all,
                       family,
                       type_modeler = 'impute',
                       bool_bias = False,
                       feat_names = None,
                       kwargs_data_val = None,
                       bool_omit_data = None):
    
    '''
    Modeler's side
    '''
                
    mask_all = np.array(mask_all_sp.todense()).reshape(-1,*X_train_all.shape)
    
    if bool_omit_data is not None:      #Sanity check
#        import ipdb;ipdb.set_trace()
        if mask_all[:,~bool_omit_data.astype(bool)].any():
            raise ValueError
    
    if kwargs_data_val is None:
        beta_est_all, c_est_all,\
        pvalues_all, tvalues_all = _estimate_masked(X_train_all, y_train_all, 
                                                   mask_all,
                                                   family = family,
                                                   type_modeler = type_modeler,
                                                   bool_bias = bool_bias,
                                                   feat_names = feat_names)
    else:
        print('using data val')
        
        beta_est_all, c_est_all,\
        pvalues_all, tvalues_all = _estimate_masked_data_val(X_train_all, y_train_all, 
                                                   mask_all,
                                                   family = family,
                                                   type_modeler = type_modeler,
                                                   bool_bias = bool_bias,
                                                   feat_names = feat_names,
                                                   **kwargs_data_val)
        
    return beta_est_all, c_est_all,\
            pvalues_all, tvalues_all
            
def _estimate_masked_data_val(X_train_all, y_train_all_0, 
                            mask_all,
                            family,
                            type_modeler,
                            bool_shuffle = False,
                            bool_bias = False,
                            feat_names = None,
                            type_data_val = 'random',
                            budget_vec = None,
                            X_val = None, 
                            y_val = None):
    
    '''
    For each mask matrix, runs the modeler
    
    In:
        mask_all:       #
    '''
    n_mask = len(mask_all)
    
    n_budget = len(budget_vec)
    
    beta_est_all = np.zeros([n_budget,n_mask,mask_all.shape[-1]])
    c_est_all = np.zeros([n_budget,n_mask])
    
    pvalues_all = np.zeros([n_budget,n_mask,mask_all.shape[-1]+1])
    tvalues_all = np.zeros([n_budget,n_mask,mask_all.shape[-1]+1])
    
    bool_d = [np.array_equal(np.unique(X_j),[0,1]) for X_j in X_train_all.T]
    idx_d = np.where(np.array(bool_d))[0]
    
    if len(idx_d)>0:
        if feat_names is None:
            print('idx_d', idx_d)
        else:
            print('idx_d', [feat_names[idx_i] for idx_i in idx_d])
    
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
        X_hat_0, y_train_all_2 = get_impute(type_modeler, 
                                        X_miss, y_train_all_1,
                                        seed = i,
                                        idx_d = idx_d)
            
        
#        data_value = helper_data_val.get_data_value(X_hat_0, y_train_all_2,
#                                                    type_data_val,
#                                                    seed = i)        
        X_all, y_all = helper_data_val.get_X_y_concat(X_hat_0, y_train_all_2,
                                                      X_val, y_val)
        
#        data_value = get_data_val_wrapped()
        data_value = helper_data_val.get_data_val_wrapped(X_all, y_all,                  
                                             bool_classify = family == 'lr',
                                             n_train = len(X_hat_0),
                                             type_data_val = type_data_val, 
                                             seed = i)
        idx_argsort = np.argsort(data_value)
        
        for ii, budget in enumerate(budget_vec):
            
            print('budget',budget)
            idx_keep = idx_argsort[budget:]
            
            X_hat = X_hat_0[idx_keep]
            y_train_all = y_train_all_2[idx_keep]
                        
        
            beta_est, c_est,\
            pvalues, tvalues = _estimate(X_hat, y_train_all,
                                        family,
                                        verbose = i == 0,
                                        bool_bias = bool_bias,
                                        return_pval = 1,
                                        feat_names = feat_names)
            
            temp_beta, temp_c = _estimate(X_hat, y_train_all,
                                          family,
                                          verbose = False,
                                          bool_bias = bool_bias)
            
            if not np.allclose(beta_est,temp_beta) or \
                not np.allclose([c_est], [temp_c]):            
    #            import ipdb;ipdb.set_trace()
                raise ValueError
        
            beta_est_all[ii,i] = beta_est
            c_est_all[ii,i] = c_est
            pvalues_all[ii,i] = pvalues
            tvalues_all[ii,i] = tvalues
        
    return beta_est_all, c_est_all, \
            pvalues_all, tvalues_all