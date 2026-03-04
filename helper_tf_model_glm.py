# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.datasets import make_spd_matrix

import os
import matplotlib.pyplot as plt
    
import helper_em_tf
#from helper_prob.models.helper_mvn_tf import get_KL

import helper_tf_glm

#import helper_dag
global count
count = 0
bool_ent = 0
#bool_sub = True
#bool_sub = False
#bool_reg = True
#bool_l1 = 1
#reg_l1 = 1

#import helper
import helper_impute_tf

class Custom(keras.Sequential):
              
    '''
    Extends the keras.Sequential for implementing a custom loss function
    '''
    
    def _set_param(self,
                   idx_input,
                   idx_mask,
                   beta_a,
                   scale_mu,
                   scale_std,
                   loss_type,
                   reg_lmbda = 0):
        
#        self.mu_a = mu_a
            
        self.date = datetime.now().strftime("_time_%H_%M_%m_%d_%Y")
        
        self.reg_lmbda = reg_lmbda
        
        self.idx_input = idx_input
        self.idx_mask = idx_mask
        
        self.beta_a = beta_a
        self.scale_mu = scale_mu
        self.scale_std = scale_std
        
        self.loss_type = loss_type
        
        p_sub = len(idx_mask)
        self.bool_o_mat = helper_impute_tf.get_obs_mat(p_sub)
                

    def train_step(self, data): 
        
        '''
        self.idx_input: determines the input
        self.idx_mask: ~determines the masked variables
        '''
                
        debug = 0
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        z = data
        global count
        if 1:
            bool_full = 1
            trainable_vars = self.trainable_variables

            plain_loss = 0
            avg_loss = 0
            
            with tf.GradientTape() as tape:
                
#                x_sub = tf.constant(x.numpy()[:,self.idx_input])                 
            
#                z_sub = tf.gather(z, indices=self.idx_input, axis = 1)                
                
                z_scaled = (z - self.scale_mu)/self.scale_std
                z_sub = tf.gather(z_scaled, indices=self.idx_input, axis = 1)
                
                p_r_x = self(z_sub, training=True)  # Forward pass                                    
                
#                import ipdb;ipdb.set_trace()
                x = tf.gather(z,indices=np.arange(z.shape[1]-1), axis = 1)
                y = tf.squeeze(tf.gather(z,indices=[z.shape[1]-1], axis = 1))
                
                beta = helper_tf_glm.glm(x, y, p_r_x, 
                                         self.idx_mask, 
                                         self.bool_o_mat)                
                
#                x_mask = tf.gather(x, indices=self.idx_mask, axis = 1)
#                mu_est = helper_impute_tf.get_mu(x_mask, p_r_x, 
#                                                 self.bool_o_mat)
                
#                print('mu_est',mu_est)
                beta = tf.gather(beta, 0, axis=1)
                beta_sel = tf.gather(beta, self.idx_mask)
                print('beta_j',np.round(beta_sel.numpy(),2))
                
                if self.loss_type == 0:                                        
                    loss = tf.norm(beta_sel)
                if self.loss_type == 1:
                    loss = tf.norm(beta-self.beta_a,ord=1)
                
#                import ipdb;ipdb.set_trace()
#                loss = beta_sel**2
                
#                loss = (mu_est - self.mu_a)**2
                #%%                                    
                plain_loss += loss.numpy()
                if bool_full:
#                        print('using reg. with',self.reg_lmbda)
                    prob_obs = helper_em_tf.get_obs_prob(p_r_x, len(self.idx_mask))
                    
                    if self.reg_lmbda != 0:
                        loss = loss + self.reg_lmbda*(1-prob_obs)
                                            
                    avg_loss += loss.numpy()

#                import ipdb;ipdb.set_trace() 
                gradients = tape.gradient(loss, trainable_vars)                
                                    
        # Update weights
        if not debug:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        else:
            print('debug:not grad')        

        if not bool_full:
            exp_obs_ratio = tf.math.reduce_mean(tf.math.reduce_mean(p_r_x,1))
            p_miss_row = tf.math.reduce_mean(tf.math.reduce_prod(1-p_r_x,1))
        else:
            exp_obs_ratio = np.nan
            p_miss_row = tf.math.reduce_mean(p_r_x[:,-1])
                        

#        n_rem, n_add = get_graph_error(K_a.numpy(), K_new.numpy()[np.newaxis,:,:], 
#                                       0.1)        
        
        exp_obs_ratio = (prob_obs*len(self.idx_mask)+x.shape[1]-len(self.idx_mask))/x.shape[1]
        
        if not bool_ent:
            avg_loss = plain_loss + self.reg_lmbda*(1-prob_obs)
            
        else:
            pass
        
        loss_dic = {'loss':loss,
                    'exp_obs_ratio':exp_obs_ratio,
                    'p_miss_row':p_miss_row,
                    'plain_loss':plain_loss,
                    'prob_obs':prob_obs,
                    'avg_loss':avg_loss,
                    'beta':beta
                    }        
            
        return loss_dic
    
    

def get_model(model_cfg, 
              idx_input,
              idx_mask,
              ):
    
    '''
    idx_input_train:  #idx of input nodes
    idx_mask:       #idx of masked nodes
        
    bool_ratio:     #experimental adding ratio as the input
    '''
    model = Custom()
    
    model.add(keras.Input(shape=(len(idx_input),)))    
        
    if model_cfg == 1:
        model.add(layers.Dense(10,activation='relu'))
    elif model_cfg == 2:
        model.add(layers.Dense(100,activation='relu'))
    elif model_cfg == 3:
        model.add(layers.Dense(10,activation='relu'))
        model.add(layers.Dense(10,activation='relu'))
    elif model_cfg == 4:
        model.add(layers.Dense(100,activation='relu'))
        model.add(layers.Dense(100,activation='relu'))            
        
    p_sub = len(idx_mask)
    
    model.add(layers.Dense(int(2**p_sub),activation='softmax'))
    
    return model

def get_savePath(model_cfg, loss_type, 
                 bool_full, mode,data_dic,
                 reg_lmbda, seed,
                 currPath,
                 bool_retrain = False,
                 mode_adv = 0,
                 bool_l1 = 0,
                 attack_node = 'pip2',
                 bool_hat_true = True,
                 bool_sub = False,
                 bool_force_zero = False,
                 bool_sim = False):
    
    dict_sub = dict(
#                corr=corr,
                model_cfg=model_cfg,
                loss_type=loss_type,
                bool_full=bool_full,
                mode=mode,
                seed=seed)
    dict_sub = {**dict_sub,**data_dic}
    
    if reg_lmbda != 0:
        dict_sub['reg_lmbda'] = reg_lmbda
    if mode_adv != 0:
        dict_sub['mode_adv'] = mode_adv
    if bool_l1 != 0:
        dict_sub['bool_l1'] = bool_l1
    if attack_node != 'pip2':
        dict_sub['attack_node'] = attack_node
    if not bool_hat_true:
        dict_sub['bool_hat_true'] = bool_hat_true
    if bool_sub:
        dict_sub['bool_sub'] = bool_sub         
    if bool_force_zero:
        dict_sub['bool_force_zero'] = bool_force_zero
    if bool_sim:
        dict_sub['bool_sim'] = bool_sim
        
    figname = '_'.join(['%s_%s'%(a,b) for a, b in dict_sub.items()])
    
    savePath = os.path.join(currPath, 'train_'+figname)

    if bool_retrain:
        savePath = os.path.join(savePath, 'retrain')
        
    return savePath
def load_model(model_cfg, loss_type, 
               bool_full, mode,data_dic,
               reg_lmbda, seed,
               currPath,
               **kwargs):
    
#    dict_sub = dict(
##                corr=corr,
#                model_cfg=model_cfg,
#                loss_type=loss_type,
#                bool_full=bool_full,
#                mode=mode,
#                seed=seed)
#    dict_sub = {**dict_sub,**data_dic}
#    
#    if reg_lmbda != 0:
#        dict_sub['reg_lmbda'] = reg_lmbda
#    if mode_adv != 0:
#        dict_sub['mode_adv'] = mode_adv
#        
#    figname = '_'.join(['%s_%s'%(a,b) for a, b in dict_sub.items()])
#    
#    savePath = os.path.join(currPath, 'train_'+figname)
#
#    if bool_retrain:
#        savePath = os.path.join(savePath, 'retrain')
    
    savePath = get_savePath(model_cfg, loss_type, 
                            bool_full, mode,data_dic,
                            reg_lmbda, seed,
                            currPath,
                            **kwargs)
    if loss_type != 9:
    #    Custom = 
        custom_objects = dict(Custom=Custom)
    else:
        custom_objects = dict(Custom2=Custom2)
        

    savePath2 = os.path.join(savePath,'model')
    
        
    model = tf.keras.models.load_model(savePath2, 
                                       custom_objects=custom_objects)

    return model