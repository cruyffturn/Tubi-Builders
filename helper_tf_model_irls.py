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
import helper_glm_wrap
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
import helper_tf_irls_ig

from helper_partial_data import OmitPortion
    
class Custom(keras.Sequential):
              
    '''
    Extends the keras.Sequential for implementing a custom loss function
    '''
    
    def _set_param(self,
                   idx_input,
                   idx_mask,
                   beta_a,
                   c_a,
                   scale_mu,
                   scale_std,
                   loss_type,
                   n_steps,
                   family,
                   bool_bias = 0,
                   reg_lmbda = 0,
                   max_steps = 1000,
                   type_modeler = 'impute',
                   glm_lmbda = 0,
                   epochs = None):
        
#        self.mu_a = mu_a
            
        self.date = datetime.now().strftime("_time_%H_%M_%m_%d_%Y")
        
        self.reg_lmbda = reg_lmbda
        
        self.idx_input = idx_input
        self.idx_mask = idx_mask
        
        self.beta_a = beta_a
        self.c_a = c_a
        
        self.scale_mu = scale_mu
        self.scale_std = scale_std
        
        self.loss_type = loss_type
        
        p_sub = len(idx_mask)
        self.bool_o_mat = helper_impute_tf.get_obs_mat(p_sub)
        
        self.n_steps = n_steps
        self.family = family
        
        self.bool_bias = bool_bias
        self.max_steps = max_steps
        
        self.type_modeler = type_modeler
        
        self.type_impute = ['mean','mice'][int(type_modeler == 'mice')]
        self.glm_lmbda = glm_lmbda
        
        self.epochs = epochs

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
                
                if self.bool_bias:
                    x_in = tf.concat([tf.ones((x.shape[0],1)),x],
                                      axis=1)
                    idx_mask_in = self.idx_mask+1
                else:
                    x_in = x
                    idx_mask_in = self.idx_mask
                    
                beta_0 = tf.constant(tf.zeros(x_in.shape[1]),
                                tf.float32) #/x_in.shape[0]
                     
                beta = helper_tf_glm.get_irls_enum(x_in, y, p_r_x, 
#                                     self.idx_mask,
                                     idx_mask_in,
                                     self.family,
                                     self.bool_o_mat,
                                     self.bool_bias,
                                     beta_0 = beta_0,
                                     n_steps = self.n_steps,
#                                     n_steps = 1,
#                                     bool_while = False
                                     bool_while = True,
                                     max_steps = self.max_steps,
                                     bool_cca = self.type_modeler == 'cca',
                                     type_impute = self.type_impute
                                     )                                 
                
                if self.bool_bias:
                    c = tf.squeeze(tf.gather(beta,indices=[0]))
                    beta = tf.gather(beta,indices=np.arange(1,x_in.shape[1]))
                else:
                    c = 0.
                    
                print(beta.numpy())
#                x_mask = tf.gather(x, indices=self.idx_mask, axis = 1)
#                mu_est = helper_impute_tf.get_mu(x_mask, p_r_x, 
#                                                 self.bool_o_mat)
                
#                print('mu_est',mu_est)                
#                beta = tf.gather(beta, 0, axis=1)
                beta_sel = tf.gather(beta, self.idx_mask)
                print('beta_j',np.round(beta_sel.numpy(),2))
                
                if self.loss_type == 0:                                        
                    loss = tf.norm(beta_sel)
                    
                elif self.loss_type == 1:
                    loss = tf.norm(beta-self.beta_a,ord=1) +\
                            tf.norm(c-self.c_a,ord=1)
                
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
        
                
        if 0:
            Z = np.concatenate([x.numpy(),y.numpy()[:,np.newaxis]],1)
            
            beta_est_all, c_est_all = helper_glm_wrap.estimate(Z, x.numpy(), y.numpy(),
                     self.scale_mu, self.scale_std,
                     self, 42,
                     self.idx_input, 
                     self.idx_mask,
                     family = self.family,
                     bool_bias = self.bool_bias,
                     n_rep = 1)[:2]
            
            import ipdb;ipdb.set_trace()
        
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
              p,
              bool_ig = False,
              bool_omit_data = None
              ):
    
    '''
    idx_input_train:  #idx of input nodes
    idx_mask:       #idx of masked nodes
        
    bool_ratio:     #experimental adding ratio as the input
    '''
    if not bool_ig:
        model = Custom()
    else:
        print('using ig custom2')
        model = Custom2()
    
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
    elif model_cfg == 5:
        model.add(layers.Dense(100,activation='relu'))
        model.add(layers.Dense(100,activation='relu'))            
        model.add(layers.Dense(100,activation='relu'))
        
    p_sub = len(idx_mask)
    
    if p_sub != p:
        n_out = int(2**p_sub)
    else:#All missing not feasible
        n_out = int(2**p_sub)-1
        import ipdb;ipdb.set_trace()
        
#    model.add(layers.Dense(int(2**p_sub),activation='softmax'))
    model.add(layers.Dense(n_out,activation='softmax'))
    
    if bool_omit_data is not None:
        model.add(OmitPortion(bool_omit_data, n_out))
        
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

class Custom2(Custom):
              
    '''
    Extends the keras.Sequential for implementing a custom loss function
    '''

    def train_step(self, data): 
        
        print('using ig')
        '''
        self.idx_input: determines the input
        self.idx_mask: ~determines the masked variables
        '''
                
        debug = 0
        bool_full = 1
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        z = data
        global count
        
        if count == 0:
            self.actual_type_modeler = self.type_modeler
        
#        n_epoch = 600
#        n_epoch = 300
        n_epoch = int(self.epochs*3/5)      #Used for fine tuning
        if count < n_epoch:
            if self.actual_type_modeler == 'mice':
                self.type_modeler = 'cca'
                print('switching to cca')
#        else:
        elif count > n_epoch:
            if self.actual_type_modeler == 'mice':
                self.type_modeler = self.actual_type_modeler
                self.optimizer.learning_rate=self.optimizer.learning_rate/100
                print('switching to back to mice')
            
        print('learning_rate', self.optimizer.learning_rate.numpy())
        gradients, p_r_x, \
        loss, plain_loss,\
        beta = helper_tf_irls_ig.get_grad(self, z,
                                          return_loss = 1)
        
        if self.bool_bias:
            beta = tf.gather(beta,indices=np.arange(1,beta.shape[0]))
                    
#        import ipdb;ipdb.set_trace()
        
        # Update weights
        if not debug:
            self.optimizer.apply_gradients(zip(gradients, 
                                               self.trainable_variables))
        else:
            print('debug:not grad')        

        if not bool_full:
            exp_obs_ratio = tf.math.reduce_mean(tf.math.reduce_mean(p_r_x,1))
            p_miss_row = tf.math.reduce_mean(tf.math.reduce_prod(1-p_r_x,1))
        else:
            exp_obs_ratio = np.nan
            p_miss_row = tf.math.reduce_mean(p_r_x[:,-1])
                        

        prob_obs = helper_em_tf.get_obs_prob(p_r_x, len(self.idx_mask))
        
        x_shape_1 = z.shape[1]-1
        
        avg_loss = -1
#        plain_loss = -1
        exp_obs_ratio = (prob_obs*len(self.idx_mask)+x_shape_1-\
                         len(self.idx_mask))/x_shape_1
        
#        avg_loss = plain_loss + self.reg_lmbda*(1-prob_obs)

        
        loss_dic = {'loss':loss,
                    'exp_obs_ratio':exp_obs_ratio,
                    'p_miss_row':p_miss_row,
                    'plain_loss':plain_loss,
                    'prob_obs':prob_obs,
                    'avg_loss':avg_loss,
                    'beta':beta
                    }        
            
        count += 1        
            
        return loss_dic
    