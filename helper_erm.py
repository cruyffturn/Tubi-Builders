'''
J:      objective function

'''
import os
import tensorflow as tf
from tensorflow import keras
#import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import helper_tf_erm_ig
from helper_partial_data import OmitPortion, MaskValidStates, get_valid_states, Mixture
#import helper_em_tf
import helper_impute_tf
#import helper_erm_nn
import helper_load_nn
#from helper_em_tf import get_outer_sum, gather_x2, powerset
#import helper_tf_model_irls
import helper_draw_nn
#from helper_erm_nn import get_flat

#from helper_erm_visualize import get_model2
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

import copy
import helper_erm_nn
from tensorflow.keras.models import clone_model
from helper_scale import _get_scale_params
#from helper_tf_irls_ig import get_reusable
#import helper_erm_nn
        
global count
count = 0
    
def train_lamm(X_train_all,
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
              family,
              model_cfg = 2,
              type_attack = None,
              bool_omit_data = None,
              kwargs_optimizer = {},
              batch_size = -1,
              shuffle=True,
              bool_partial_read = False,
              **kwargs,
              ):
    
        '''        
        Trains the NN For attacking a inner solver
        '''
#        import ipdb;ipdb.set_trace()        
            
            #scaler = StandardScaler()
        scale_mu = np.mean(X_train_all,0)
        scale_std = np.std(X_train_all,0)
        
        scale_std[scale_std==0] = 1
        
        if not bool_scale_all:
            mu_y = 0
            std_y = 1
        else:
            mu_y = np.mean(y_train_all,0)
            std_y = np.std(y_train_all,0)
            print('scaling all')
            
        scale_mu = np.append(scale_mu,[mu_y])
        scale_std = np.append(scale_std,[std_y])

        if 'solver_arch' in kwargs['kwargs_solver'].keys():
            if kwargs['kwargs_solver']['solver_arch'] == 'rnn_p2012_all':
                scale_mu = np.zeros_like(scale_mu)
                scale_std = np.ones_like(scale_std)
        
        tf.random.set_seed(seed_model)        
            
        #idx_adv = range(p)
        p = X_train_all.shape[1]
        
        if model_cfg >= 0:
#            shuffle=True            
            model, n_out = get_model(model_cfg, 
                                     idx_input,
                                     idx_mask,
                                     p,                                        
                                     bool_omit_data = bool_omit_data,
                                     bool_partial_read = bool_partial_read,
                                     X_train_all = X_train_all
                                     )
        else:
            shuffle=False
            model, n_out = get_model_fixed(idx_mask, p, 
                                           len(X_train_all),
                                           model_cfg,
                                           bool_omit_data = bool_omit_data,
                                           bool_partial_read = bool_partial_read
                                           )
            if model_cfg == -1:
                model.set_weights([get_init_weight(X_train_all, y_train_all,
                                                   X_target, y_target)])
    
            elif model_cfg == -3:
                model.set_weights([get_init_weight2(X_train_all)])
            
#            model, n_out = get_model2(model_cfg, 
#                                 idx_input,
#                                 idx_mask,
#                                 p,                                        
#                                 bool_omit_data = bool_omit_data
#                                 )                   
            
        model._set_param(idx_input,
                         idx_mask,
                         X_target,
                         y_target,
                         n_out,
                         scale_mu=scale_mu,
                         scale_std=scale_std,
                         epochs = epochs,
                         family = family,
                         **kwargs)
        if model.type_scaler is not None:
            model._set_scaler(X_train_all)

        if model.loss_type == 3:
            model.model_adv = get_ta(model, X_train_all, y_train_all)
            
#        import ipdb;ipdb.set_trace()
        if kwargs_optimizer != {}:
            print('using',kwargs_optimizer)
            
        if name_optimizer == 'adam':
            optim = keras.optimizers.Adam(learning_rate=lr,
                                          **kwargs_optimizer
                                          )
            
        elif name_optimizer == 'adagrad':
            optim = keras.optimizers.Adagrad(learning_rate=lr,
                                             **kwargs_optimizer)
            
        elif name_optimizer == 'SGD':
            optim = keras.optimizers.SGD(learning_rate=lr,
                                         **kwargs_optimizer)
                    
        model.compile(
        #              optimizer=keras.optimizers.Adam(lr),
                      optimizer=optim,
                      run_eagerly = 1,
                      )
                
#        import ipdb;ipdb.set_trace()
#        min_delta = 1e-4
        callbacks = []
#             tf.keras.callbacks.LearningRateScheduler(scheduler)]        
        
        Z = np.concatenate([X_train_all,y_train_all[:,np.newaxis]],1)
        
        if batch_size == -1:
            batch_size = len(Z)
        
        #batch_size has to be all samples for bool_omit_data to work
        history = model.fit(Z, 
                            epochs=epochs,
#                            batch_size=len(Z),
                            batch_size=batch_size,
                            callbacks=callbacks,
                            shuffle=shuffle
                            )
        
        return model, history, scale_mu, scale_std


def get_model(model_cfg, 
              idx_input,
              idx_mask,
              p,
              bool_omit_data = None,
              bool_partial_read = False,
              X_train_all = None,
              ):
    
    '''
    idx_input_train:  #idx of input nodes
    idx_mask:       #idx of masked nodes
        
    bool_ratio:     #experimental adding ratio as the input
    '''
    model = BLAMM()
    
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
    elif model_cfg == 6:
        pass
        
    p_sub = len(idx_mask)
    
    if p_sub != p:
        n_out = int(2**p_sub)
    else:#All missing not feasible
        n_out = int(2**p_sub)-1
        import ipdb;ipdb.set_trace()
        
#    model.add(layers.Dense(int(2**p_sub),activation='softmax'))
    model.add(layers.Dense(n_out,activation='softmax'))
    
    if bool_omit_data is not None:
        if not bool_partial_read:
            print('using partial omission')
            model.add(OmitPortion(bool_omit_data, n_out))
        else:
            print('using partial read')
            model.add(Mixture(bool_omit_data, n_out))
        
    if np.isnan(X_train_all[:,idx_mask]).any():
        import ipdb;ipdb.set_trace()
        print('Target data already has missingness')
        bool_o_mat = helper_impute_tf.get_obs_mat_wrap(p_sub, n_out)
        valid_states = get_valid_states(X_train_all, idx_mask, 
                                        bool_o_mat)
        model.add(MaskValidStates(valid_states))
        
    return model, n_out

class BLAMM(keras.Sequential):
              
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
                   family,
                   reg_lmbda = 0,
                   type_modeler = 'impute',
                   epochs = None,
                   kwargs_solver = {'epochs':500},
                   fig_path = None,
                   n_solvers = 1,
                   method = 'inv',
                   n_return_states = 10,
                   loss_type = 0,
                   type_scaler = None,
                   bool_resolve = True
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
        
        self.n_solvers = n_solvers
        self.method = method
        self.n_return_states = n_return_states
        
        self.loss_type = loss_type     
        self.family = family
        
        self.type_scaler = type_scaler
        self.bool_resolve = bool_resolve
    
    def _set_scaler(self, X):
        
        self.type_scaler = _get_scale_params(X, self.type_scaler)
        
    def train_step(self, data): 
        
#        print('using ig')
        '''
        self.idx_input: determines the input
        self.idx_mask: ~determines the masked variables
        '''
        
        if self.type_scaler is not None:
            if type(self.type_scaler) is str:
                raise ValueError
                
        debug = 0
        debug_grad = 0
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
        
        if self.n_solvers > 1:
            raise ValueError
            
        for i in range(self.n_solvers):
            
            if self.method == 'inv':
#                gradients_i, p_r_x, \
#                inner_model, metrics_i = helper_tf_erm_ig.get_grad(self, z,
#                                                                   return_loss = 1)            
                gradients_i, p_r_x, \
                inner_model, metrics_i = helper_tf_erm_ig.get_grad_inv(self, z, 
                                                                       return_loss = 1)
                
                if debug_grad:
                    grad_check = helper_tf_erm_ig.get_grad(self, z,
                                                           return_loss = 1)
                        
                    for grad_1, grad_2 in zip(gradients_i, grad_check):
                        if not np.allclose(grad_1,grad_2,atol=1e-1):
                            print(np.abs(grad_1-grad_2).max())            
                            import ipdb;ipdb.set_trace()
                            
                
            elif self.method == 'rev':
                gradients_i, p_r_x, \
                inner_model, metrics_i = helper_tf_erm_ig.get_grad_rmd(self, z,
                                                                   return_loss = 1)
            
            elif self.method == 'unroll':
                gradients_i, p_r_x, \
                inner_model, metrics_i = helper_tf_erm_ig.get_grad_unroll(self, z,
                                                                   return_loss = 1)
                if debug_grad:
                    grad_check = helper_tf_erm_ig.get_grad_rmd(self, z,
                                                               return_loss = 1)[0]
                    for grad_1, grad_2 in zip(gradients_i, grad_check):
                        if not np.allclose(grad_1,grad_2,atol=1e-6):
                            print(np.abs(grad_1-grad_2).max())            
                            import ipdb;ipdb.set_trace()
                            
            elif self.method == 'pen':
                gradients_i, p_r_x, \
                inner_model, metrics_i = helper_tf_erm_ig.get_grad_pen(self, z,
                                                                       return_loss = 1)


            
            
            if i == 0:
                gradients = gradients_i
                metrics = metrics_i
            else:
                for i in range(len(gradients)):
                    gradients[i] = gradients[i] + gradients_i[i]
                
                for key, val in metrics_i.items():
                    metrics[key] = metrics[key] + val
                
        #Averaging
        if self.n_solvers != 1:
            for i in range(len(gradients)):
                gradients[i] = gradients[i]/self.n_solvers
                
            for key in metrics.keys():
                metrics[key] = metrics[key]/self.n_solvers
            
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
        
        #%% Class specific
        y = tf.squeeze(tf.gather(z,indices=[z.shape[1]-1], axis = 1))
        
        p_r_x_1 = tf.boolean_mask(p_r_x,tf.cast(y,tf.bool))
        p_r_y1 = helper_load_nn.get_obs_prob_wrap(p_r_x_1, len(self.idx_mask),
                                                 self.n_out)
        
        p_r_x_0 = tf.boolean_mask(p_r_x, ~tf.cast(y,tf.bool))
        p_r_y0 = helper_load_nn.get_obs_prob_wrap(p_r_x_0, len(self.idx_mask),
                                                 self.n_out)
        
        #%%
#        prob_obs = helper_em_tf.get_obs_prob(p_r_x, len(self.idx_mask))
        
        x_shape_1 = z.shape[1]-1
        
#        avg_loss = -1
#        plain_loss = -1
#        exp_obs_ratio = (prob_obs*len(self.idx_mask)+x_shape_1-\
#                         len(self.idx_mask))/x_shape_1

        #%%
        if self.loss_type in [0,3]:            
            _target_score = tf.math.sigmoid(inner_model(self.X_target, 
                                                         training=False))
            target_score = _target_score*self.y_target + \
                            (1-_target_score)*(1-self.y_target)
            target_score = tf.squeeze(target_score)
            metrics_other = dict(target_score=target_score)
        else:
            metrics_other = {}
            
#            print('label',self.y_target,'prob',tf.math.sigmoid(y_pred))
        #%%
        loss_dic = {
#                    'exp_obs_ratio':exp_obs_ratio,
#                    'p_miss_row':p_miss_row,
                    'prob_obs':prob_obs,
                    'p_r_y1':p_r_y1,
                    'p_r_y0':p_r_y0,
#                    'avg_loss':avg_loss,
#                    'beta':beta
                    }
        loss_dic.update(metrics_other)
        loss_dic.update(metrics)                
        
#        if self.X_target.shape[1] == 2:
        if (count % max(int(self.epochs/10),1) == 0) or (count == self.epochs-1):
            if self.fig_path is not None:
                
                if not os.path.exists( self.fig_path):
                    os.makedirs( self.fig_path)

                fig, axs = plt.subplots() 
#                    fig, axs = plt.subplots(1,3)                    
                s = p_r_x.numpy()[:,0]

                
                if self.X_target.shape[1] == 2:
                    
                    # Normalize the values between 0 and 1
                    min_size = 10
                    max_size = 200                
    
                    norm_s = (s - 0) / (1 - np.min(s))
                    marker_sizes = norm_s * (max_size - min_size) + min_size


                    x_min, x_max, \
                    y_min, y_max = helper_draw_nn.draw_decision_boundary(z.numpy()[:,:-1], 
                                                                         z.numpy()[:,-1], 
                                                                         self.X_target, 
                                                                         self.y_target,
                                                                     inner_model, 
                                                                     axs,
                                                                     s=marker_sizes)
                else:
                    min_size = 1
                    max_size = 10
    
                    norm_s = (s - 0) / (1 - np.min(s))
                    marker_sizes = norm_s * (max_size - min_size) + min_size


                    x_min, x_max, \
                    y_min, y_max = helper_draw_nn.draw_decision_boundary_pca(z.numpy()[::100,:-1], 
                                                                         z.numpy()[::100,-1], 
                                                                         self.X_target, 
                                                                         self.y_target,
                                                                     inner_model, 
                                                                     axs,
                                                                     s=marker_sizes[::100])
                    
                
                if 0:
                    for y_in, ax in enumerate(axs[1:]):
                        
                        helper_draw_nn.draw_decision_boundary2(z.numpy()[:,:-1], 
                                                               z.numpy()[:,-1], 
                                                               self.X_target, 
                                                               self.y_target,
                                                               self, ax,
                                                               y_in,
                                                               x_min, x_max,
                                                               y_min, y_max)
                
                fig.set_size_inches( w = 5, h = 5)
#                    fig.set_size_inches( w = 15, h = 5)
                fig.savefig(os.path.join(self.fig_path,'fig_%i.png'%count), 
                            dpi=200, bbox_inches='tight')
                plt.close('all')

        #%%
        count += 1        
            
        return loss_dic
    
class Sigmoid(keras.layers.Layer):
    def __init__(self, N, n_out_0):
        super().__init__()
        
        if n_out_0 == 2:
            n_out = 1
        else:
            n_out = n_out_0
            
        self.w = self.add_weight(
            shape=(N, n_out),
#            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1),
#            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=.1),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        self.tree = None
        
#        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs, training = False):
                
        if training:
            weight = self.w            
            
            if self.tree is None:
#                import ipdb;ipdb.set_trace()
                try:
                    self.tree = KDTree(inputs.numpy())
                except:
                    print('KDTree not working')
                    self.tree = -1
        else:
            print('testing')
            if self.tree != -1:
                _, idx = self.tree.query(inputs.numpy(), k=1)
                weight = self.w.numpy()[idx]
            else:
                print('KDTree not working')
                weight = self.w
        
        if weight.shape[1] > 1:
            return tf.nn.softmax(weight)
        elif weight.shape[1] == 1:
            y_0 = tf.nn.sigmoid(weight)
            y = tf.concat([y_0,1-y_0],1)
            return y

class SigmoidDrop(Sigmoid):
            
    def call(self, inputs, training = False):
                
        if training:        
            rand_tensor = tf.random.uniform(self.w.shape, minval=0, maxval=1)
            threshold = 0.9
            # Create the binary mask by comparing with the threshold
            mask = tf.cast(rand_tensor < threshold, dtype=tf.float32)
            
#            import ipdb;ipdb.set_trace()

            weight = self.w*(mask) + self.w.numpy()*(1-mask)
            
            if self.tree is None:
                self.tree = KDTree(inputs.numpy())
        else:
            print('testing')    
            _, idx = self.tree.query(inputs.numpy(), k=1)
            weight = self.w.numpy()[idx]
        
        if weight.shape[1] > 1:
            return tf.nn.softmax(weight)
        elif weight.shape[1] == 1:
            y_0 = tf.nn.sigmoid(weight)
            y = tf.concat([y_0,1-y_0],1)
            return y
            
  
def get_model_fixed(idx_mask, p, N, model_cfg,
                    bool_omit_data,
                    bool_partial_read):
    
    p_sub = len(idx_mask)
    
    if p_sub != p:
        n_out = int(2**p_sub)
    else:#All missing not feasible
        n_out = int(2**p_sub)-1
        import ipdb;ipdb.set_trace()
        
    model = BLAMM()
    if model_cfg != -5:
        model.add(Sigmoid(N,n_out))
    else:
        model.add(SigmoidDrop(N,n_out))
        
    if bool_omit_data is not None:
        if not bool_partial_read:
            print('using partial omission')
            model.add(OmitPortion(bool_omit_data, n_out))
        else:
            if model_cfg != -20:
                print('using partial read')
                model.add(Mixture(bool_omit_data, n_out))
            else:
                print('Debugging partial read')

    model.bool_fixed = True
    
    return model, n_out

def get_init_weight(X_train_all, y_train_all,
                    X_target, y_target):

#    metric = 'cosine'
    k = 25
    metric = 'euclidean'
#    metric = 'cosine'

    bool_y = y_train_all == y_target
    dist = cdist(X_train_all[bool_y], X_target,
                 metric=metric)[:,0]
    
    closest = np.argsort(dist)[:k]
    
    bool_discard = np.zeros(len(X_train_all),bool)
    bool_discard[np.where(bool_y)[0][closest]] = 1
        
    weight = np.zeros([len(y_train_all),1])
    weight[bool_discard] = -5
    weight[~bool_discard] = 5
    
    return weight

def get_init_weight2(X_train_all):

#    metric = 'cosine'
#    k = 25
#    metric = 'euclidean'
#    metric = 'cosine'

    #    bool_discard = (y_train_all & (X_train_all[:,1]<1)) |\
#                    (~y_train_all & (X_train_all[:,1]>1))
    bool_discard = np.zeros(len(X_train_all),bool)
    bool_discard[np.where(np.abs(X_train_all[:,1])<1)[0]] = 1
        
    weight = np.zeros([len(X_train_all),1])
    weight[bool_discard] = -5
    weight[~bool_discard] = 5
    
    return weight

def get_ta(self, X_train_all, y_train_all):
    
    temp_kwargs = copy.deepcopy(self.kwargs_solver)
#    temp_kwargs['solver_epochs'] = 300
#    temp_kwargs['solver_optimizer'] = 'adam'
#    temp_kwargs['solver_lr'] = 1e-1
    
    if 'solver_optimizer_kwargs' in temp_kwargs.keys():
        temp_kwargs.pop('solver_optimizer_kwargs')        
    
    temp_p_r_x = np.stack([np.ones(len(X_train_all)),
                           np.zeros(len(X_train_all))],1).astype(np.float32)
    model, _ = helper_erm_nn._solve_y(X_train_all, y_train_all, 
                                            temp_p_r_x,
                                            idx_mask_in = np.array([0]),
                                            type_impute = None, 
                                            type_modeler = 'cca',
                                            kwargs_imp = {},
                                            **temp_kwargs)
        
    _unlearn(model,
             X_train_all,
             self.X_target, self.y_target,
#             lr = 1e-2,
#             lmbda = 5e-2,
             lr = 1e-3,    
             lmbda = 1,
             n_iter = 300)
#    import ipdb;ipdb.set_trace()
    fig, axs = plt.subplots()
            
    if self.X_target.shape[1] == 2:        

        x_min, x_max, \
        y_min, y_max = helper_draw_nn.draw_decision_boundary(X_train_all, 
                                                             y_train_all, 
                                                             self.X_target, 
                                                             self.y_target,
                                                         model, 
                                                         axs,
#                                                             s=marker_sizes
                                                         )
            
        fig.set_size_inches( w = 5, h = 5)
#                    fig.set_size_inches( w = 15, h = 5)
        os.makedirs( self.fig_path, exist_ok = True)
        fig.savefig(os.path.join(self.fig_path,'ta_fig_.png'), 
                    dpi=200, bbox_inches='tight')
        plt.close('all')
    return model

def _unlearn(model,
             X_train_all,
             X_target, y_target,
             lr,
             lmbda,
             n_iter):
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model_0 = clone_model(model)
    model_0.set_weights(model.get_weights())

    class Temp():    
        def __init__(self):
            self.loss_type = 0
            self.X_target = X_target
            self.y_target = y_target
        
#    family = 'clsf'
    temp = Temp()
    
    for i in range(n_iter):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            
            loss_0 = helper_tf_erm_ig._get_loss(temp, model, X_train_all)
            reg = tf.reduce_mean((model_0(X_train_all)-model(X_train_all))**2)
            
            loss = loss_0 + lmbda*reg    
        
        grad = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grad, 
                                      model.trainable_variables))
        
        prob = tf.squeeze(tf.math.sigmoid(model(X_target)))
#        acc, auc = _evaluate(model, x_in, y, family)
        print('%.2f,%.2f,%.2f'%(loss_0,reg,prob))    