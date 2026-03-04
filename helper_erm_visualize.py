# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow import keras
#import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import helper_tf_erm_ig
from helper_partial_data import OmitPortion
#import helper_em_tf
import helper_impute_tf
#import helper_erm_nn
import helper_load_nn
#from helper_em_tf import get_outer_sum, gather_x2, powerset
#from helper_tf_glm import get_impute, get_impute_reusable
#import helper_tf_model_irls
import helper_draw_nn
from helper_erm_nn import get_flat

global count
count = 0

class BLAMM2(keras.Model):
              
    '''
    Extends the keras.Sequential for implementing a custom loss function
    '''
    def _set_param(self,
                   idx_input,
                   idx_mask,
                   X_target,
                   y_target,
                   n_out,
                   scale_mu,
                   scale_std,
                   reg_lmbda = 0,
                   type_modeler = 'impute',
                   epochs = None,
                   kwargs_solver = {'epochs':500},
                   fig_path = None
                   ):
         
        self.reg_lmbda = reg_lmbda
        
        self.idx_input = idx_input
        self.idx_mask = idx_mask
                
        self.scale_mu = scale_mu
        self.scale_std = scale_std    
        
        p_sub = len(idx_mask)
        self.n_out = n_out
        self.bool_o_mat = helper_impute_tf.get_obs_mat_wrap(p_sub, n_out)
        
        self.type_modeler = type_modeler
        
        self.type_impute = ['mean','mice'][int(type_modeler == 'mice')]        
        self.epochs = epochs    
        
        self.X_target = X_target
        self.y_target = y_target
        
        self.kwargs_solver = kwargs_solver
        self.bool_bias = False #Legacy
        
        self.prev_solution = None #initializing for warm start
        
        self.fig_path = fig_path
        
    def train_step(self, data): 
        
#        print('using ig')
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
        inner_model, metrics = helper_tf_erm_ig.get_grad(self, z,
                                                return_loss = 1)
        
        print('outer_grad_norm',
              np.linalg.norm(gradients[0].numpy().ravel(),1))
                  
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
                        

        prob_obs = helper_load_nn.get_obs_prob_wrap(p_r_x, len(self.idx_mask),
                                                   self.n_out)
#        prob_obs = helper_em_tf.get_obs_prob(p_r_x, len(self.idx_mask))
        
        x_shape_1 = z.shape[1]-1
        
        avg_loss = -1
#        plain_loss = -1
        exp_obs_ratio = (prob_obs*len(self.idx_mask)+x_shape_1-\
                         len(self.idx_mask))/x_shape_1
        
#        avg_loss = plain_loss + self.reg_lmbda*(1-prob_obs)

        
        loss_dic = {
                    'exp_obs_ratio':exp_obs_ratio,
                    'p_miss_row':p_miss_row,
                    'prob_obs':prob_obs,
                    'avg_loss':avg_loss,
#                    'beta':beta
                    }
        loss_dic.update(metrics)
        
        if (count % 10 == 0) or (count == self.epochs-1):
            if self.fig_path is not None:
                self.draw_boundary(z, inner_model, count)


        #%%
        count += 1        
            
        return loss_dic
    
    
    def draw_boundary(self, z, inner_model, 
                      count, title=''):
    
        if not os.path.exists( self.fig_path):
            os.makedirs( self.fig_path)
    
        fig, axs = plt.subplots(1,3)                    
        x_min, x_max, \
        y_min, y_max = helper_draw_nn.draw_decision_boundary(z.numpy()[:,:-1], 
                                                             z.numpy()[:,-1], 
                                                             self.X_target, 
                                                             self.y_target,
                                                         inner_model, 
                                                         axs[0])
        
        for y_in, ax in enumerate(axs[1:]):
            
            helper_draw_nn.draw_decision_boundary2(z.numpy()[:,:-1], 
                                                   z.numpy()[:,-1], 
                                                   self.X_target, 
                                                   self.y_target,
                                                   self, ax,
                                                   y_in,
                                                   x_min, x_max,
                                                   y_min, y_max)
        
        if title != '':
            axs[0].set_title(title)
            
        fig.set_size_inches( w = 15, h = 5)
        fig.savefig(os.path.join(self.fig_path,'fig_%i.png'%count), 
                    dpi=200, bbox_inches='tight')
        plt.close('all')
                    
def get_model2(model_cfg, 
              idx_input,
              idx_mask,
              p,
              bool_omit_data = None
              ):
    
    '''
    idx_input_train:  #idx of input nodes
    idx_mask:       #idx of masked nodes
        
    bool_ratio:     #experimental adding ratio as the input
    '''
#    model = BLAMM2()
        
    p_sub = len(idx_mask)
    
    if p_sub != p:
        n_out = int(2**p_sub)
    else:#All missing not feasible
        n_out = int(2**p_sub)-1
        import ipdb;ipdb.set_trace()

    inputs = layers.Input(shape=(len(idx_input),))
    
#    import ipdb;ipdb.set_trace()
    x_all = tf.gather(inputs,indices=np.arange(len(idx_input)-1), axis = 1)
    y_all = tf.gather(inputs,indices=[len(idx_input)-1], axis = 1)
#    x_all = tf.gather(inputs,indices=np.arange(len(idx_input)-1), axis = 1)
    
#    condition_mask = tf.cast(y_all, tf.int32)
#    x_l = tf.dynamic_partition(x, condition_mask, 2)
        
#    import ipdb;ipdb.set_trace()
    if model_cfg == -1:
        
        prob_l = []
        for i in range(2):
    #        z = layers.Dense(10,activation='relu')(x_all)
            z = x_all
            prob_l.append(layers.Dense(n_out,activation='softmax')(z))

        prob = prob_l[0]*y_all + prob_l[1]*(1-y_all)
    else:
        z = x_all
        z_sub = tf.gather(z,indices=[0], axis = 1)
#        prob_0 = layers.Dense(n_out,activation='softmax',
#                              kernel_initializer='random_normal')(z_sub)
#        prob_1 = prob_0
        
#        prob_0 = layers.Dense(1,activation='sigmoid',
#                              kernel_initializer='random_normal')(z_sub)
        prob_0 = layers.Activation(keras.activations.sigmoid)(FixedKernelDense(1)(z_sub))
        
        prob_1 = tf.concat([prob_0,1-prob_0],1)
            
        prob = tf.concat([tf.ones_like(y_all),tf.zeros_like(y_all)],1)*y_all \
                + prob_1*(1-y_all)
#        import ipdb;ipdb.set_trace()
        
#    condition_indices = tf.dynamic_partition(tf.range(tf.shape(x_all)[0]), 
#                                             condition_mask, 2)
#    output = tf.dynamic_stitch(condition_indices, out_l)
    
    #%%    
    if bool_omit_data is not None:
        prob = OmitPortion(bool_omit_data, n_out)(prob)
        import ipdb;ipdb.set_trace()
        
    model = BLAMM2(inputs=inputs, outputs=[prob])
    return model, n_out

def visualize(X_train_all,
              y_train_all,
              idx_input,
              idx_mask,
              X_target,
              y_target,
              seed_model,
              lr,
              epochs,
              name_optimizer,
              bool_scale_all,
              model_cfg = 2,
              type_attack = None,
              bool_omit_data = None,
              **kwargs,
              ):
    
        '''        
        Trains the NN For attacking a inner solver
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

        
        tf.random.set_seed(seed_model)        
            
        #idx_adv = range(p)
        p = X_train_all.shape[1]
        
        if model_cfg not in [-1,-2]:
            model, n_out = get_model(model_cfg, 
                                     idx_input,
                                     idx_mask,
                                     p,                                        
                                     bool_omit_data = bool_omit_data
                                     )                   
        else:
            model, n_out = get_model2(model_cfg, 
                                 idx_input,
                                 idx_mask,
                                 p,                                        
                                 bool_omit_data = bool_omit_data
                                 )                   
            
        model._set_param(idx_input,
                         idx_mask,
                         X_target,
                         y_target,
                         n_out,
                         scale_mu=scale_mu,
                         scale_std=scale_std,
                         epochs = epochs,
                         **kwargs)
        
                
        Z = np.concatenate([X_train_all,y_train_all[:,np.newaxis]],1)

#        b_vec = np.arange(0,1,0.01)
        b_vec = np.arange(0.6,1,0.01)
        grad_l = []
        for i, b in enumerate(b_vec):
#            print('w',w,'b',b)
#            model.get_layers('dense').set_weights
#            import ipdb;ipdb.set_trace()
            temp = model.get_layer('fixed_kernel_dense')
            temp.set_weights([np.array([b], np.float32),
                              temp.get_weights()[1]
                              ])
#            model.get_layer('dense').set_weights([np.array([[w]], np.float32), 
#                                                 np.array([b], np.float32)])
    
            gradients, inner_model, metrics = _visualize(model, 
                                                         Z.astype(np.float32))
            
            model.draw_boundary(tf.constant(Z.astype(np.float32)), inner_model, i,
                                title='loss=%.3f'%metrics['plain_loss'])
            
#            import ipdb;ipdb.set_trace()
            grad_l.append(gradients[0].numpy()[0])
            if i == 0:
                result_dict = {key: [val] for key,val in metrics.items()}
            else:
                for key in result_dict.keys():
                    result_dict[key].append(metrics[key])

#            loss_l.append(plain_loss)
                    
        grad_grid = np.stack(grad_l)
        for key in result_dict.keys():
            result_dict[key] = np.array(result_dict[key])
            
        return b_vec, \
                grad_grid, \
                result_dict            
    
def _visualize(self, data): 
                        
    z = data
    gradients, p_r_x, \
    inner_model, metrics = helper_tf_erm_ig.get_grad(self, z,
                                                     return_loss = 1)
    
    return gradients, inner_model, metrics

class FixedKernelDense(layers.Layer):
    def __init__(self, units=1, **kwargs):
        super(FixedKernelDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
#        print('100')
        # Create kernel with trainable=False and initialize it to constant 1
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(10.0),
            trainable=False  # Fixed kernel, not updated during training
        )
        # Create bias with trainable=True (default behavior)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',  # or any other initializer you choose
            trainable=True
        )
        super(FixedKernelDense, self).build(input_shape)

    def call(self, inputs):
        # Perform the forward computation: kernel is fixed while bias is trainable.
#        return tf.matmul(inputs, self.kernel) + self.bias
        return self.kernel*(inputs + self.bias)

    def get_config(self):
        config = super(FixedKernelDense, self).get_config()
        config.update({'units': self.units})
        return config
