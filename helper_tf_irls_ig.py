# -*- coding: utf-8 -*-
'''
Optimizing the weights
'''

import numpy as np
import copy
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import helper_tf_glm
import helper_em_tf
#%%
#import helper_dag
global count
count = 0

def get_loss(self, beta_0, z_sub,
             x_in, y,
             return_all = False):
    
    bool_full = 1
    p_r_x = self(z_sub, training=True)  # Forward pass                                    
        
    if self.bool_bias:
        c = tf.squeeze(tf.gather(beta_0,indices=[0]))
        beta = tf.gather(beta_0,indices=np.arange(1,beta_0.shape[0]))
    else:        
        c = 0.
        beta = beta_0
            
    beta_sel = tf.gather(beta, self.idx_mask)
    print('beta_j',np.round(beta_sel.numpy(),2))
    
    if self.loss_type == 0:                                        
        loss = tf.norm(beta_sel)
        
    elif self.loss_type == 1:
        loss = tf.norm(beta-self.beta_a,ord=1) +\
                       tf.norm(c-self.c_a,ord=1)
                       
    elif self.loss_type == 2:
        
        print('using KL loss')
        if self.bool_bias:
            beta_a_in = tf.concat([[self.c_a], self.beta_a],0)
            beta_a_in = tf.cast(beta_a_in,tf.float32)
        else:
            beta_a_in = self.beta_a
            
        loss = helper_tf_glm.get_kl(x_in, beta_a_in, 
                                    beta_0, self.family,
                                    y = y)
    
    #%%                                    
    plain_loss = loss.numpy()
    
    if bool_full:
        prob_obs = helper_em_tf.get_obs_prob(p_r_x, len(self.idx_mask))
        
        if self.reg_lmbda != 0:
            loss = loss + self.reg_lmbda*(1-prob_obs)                                
    
    if not return_all:
        return loss
    else:
        return loss, plain_loss
                            
def get_f_y_fyy(self, beta, z_sub, 
                x_in, y,
                idx_mask_in):
    
    '''
    following https://www.tensorflow.org/api_docs/python/tf/GradientTape#gradient
    '''
#    f_y = -2*(y-x)
#    f_yy = -2*tf.ones(len(y))
#    import ipdb;ipdb.set_trace()
    
    p_r_x = self(z_sub, training=True)  # Forward pass                                    
        
    f_y, f_yy = helper_tf_glm.sub_irls_enum(x_in, y,
                                          beta,
                                          p_r_x, 
                                          idx_mask_in,
                                          self.bool_o_mat,
                                          self.family,
                                          self.bool_bias,
                                          type_impute = self.type_impute,
                                          bool_cca = self.type_modeler == 'cca',
                                          return_grad = True,
                                          lmbda = self.glm_lmbda
                                          )    
        
    return f_y, f_yy
    
def solve_y(self, z_sub, 
            x_in, y,
            idx_mask_in):
    
    p_r_x = self(z_sub, training=True)  # Forward pass                                    
        
    beta_0 = tf.constant(tf.zeros(x_in.shape[1]),
                         tf.float32) #/x_in.shape[0]


    beta = helper_tf_glm.get_irls_enum(x_in, y, 
                                       p_r_x,     
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
                                         bool_solve_np = True,
                                         type_impute = self.type_impute,
                                         lmbda = self.glm_lmbda
                                         )
    
    return beta

def get_reusable(self, z):

    #Preliminary
    z_scaled = (z - self.scale_mu)/self.scale_std
    z_sub = tf.gather(z_scaled, indices=self.idx_input, axis = 1)
                                
    x = tf.gather(z,indices=np.arange(z.shape[1]-1), axis = 1)
    y = tf.squeeze(tf.gather(z,indices=[z.shape[1]-1], axis = 1))
    
    if self.bool_bias:
        x_in = tf.concat([tf.ones((x.shape[0],1)),x],
                          axis=1)
        idx_mask_in = self.idx_mask+1
    else:
        x_in = x
        idx_mask_in = self.idx_mask
    
    return z_sub, x_in, y, idx_mask_in

def get_grad(self, z,
             return_loss = False):
    
#    import ipdb;ipdb.set_trace()
    z_sub, x_in, \
    y, idx_mask_in = get_reusable(self, z)
    
    beta = solve_y(self, z_sub, 
                   x_in, y,
                   idx_mask_in)
        
    
    #Calculates the gradients of the outer objective
    with tf.GradientTape() as tape:        
        tape.watch([beta])
                                                            
        #Calculates the loss
        loss, plain_loss = get_loss(self, beta, z_sub, 
                                    x_in, return_all = 1,
                                    y = y)
        
        all_vars = self.trainable_variables + [beta]
        grad_all = tape.gradient(loss, all_vars)
        
    grad_x_1_l = grad_all[:-1]
    grad_beta = grad_all[-1]
    
    #Calculates the derivatives of the inner objective
    with tf.GradientTape() as tape:        
        _g, _H = get_f_y_fyy(self, beta, 
                             z_sub, x_in, 
                             y, idx_mask_in)
    
    #Calculates f_yx
    f_x_l = tape.jacobian(_g, self.trainable_variables)
            
    bool_inv = 0
#    import ipdb;ipdb.set_trace()
    if bool_inv:
#        H_inv = np.linalg.inv(_H.numpy())
#        H_inv = tf.constant(H_inv)
        H_inv = tf.linalg.inv(_H)
        A1 = tf.einsum('ij,j->i', H_inv, grad_beta)
        grad_x_2_l = []
        
        for f_x in f_x_l:
            shp = f_x.shape[1:]
            temp = tf.reshape(f_x,(f_x.shape[0],-1))
            temp2 = tf.einsum('ij,i->j', temp, A1)
            A2 = tf.reshape(temp2, shp)
            A2 = -A2
            grad_x_2_l.append(A2)
    else:
        grad_x_2_l = []
        
        for f_x in f_x_l:
            
            shp = f_x.shape[1:]
            temp = tf.reshape(f_x,(f_x.shape[0],-1))
            if 0:
                B1 = tf.linalg.lstsq(_H,temp)   #n_beta,n_x                
            else:
                B1 = np.linalg.lstsq(_H.numpy(),temp.numpy())[0]
                B1 = tf.constant(B1)
                
            B2 = tf.einsum('ij,i->j', B1, grad_beta)    #n_x                
            A2 = tf.reshape(B2, shp)
            A2 = -A2            
                
            grad_x_2_l.append(A2)
    
#        [np.abs(A2-A3).max() for A2,A3 in zip(grad_x_2_l,A3_l)]
            
#    import ipdb;ipdb.set_trace()
    temp_iter = zip(grad_x_1_l,grad_x_2_l)
    grad = [grad_x_1+grad_x_2 for grad_x_1,grad_x_2 in temp_iter] 
            
    if not return_loss:
        return grad
    else:
        p_r_x = self(z_sub, training=True)  # Forward pass
        
        temp_check = self(z_sub, training=False)
        if np.any(temp_check.numpy() != p_r_x.numpy()):
            print('dropout can cause a problem since p_r_x called multiple')
            
        return grad, p_r_x, \
                loss, plain_loss,\
                beta
        
