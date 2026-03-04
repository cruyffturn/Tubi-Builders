# -*- coding: utf-8 -*-
'''
Optimizing the weights
'''

import numpy as np
import copy

import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers

#import helper_tf_glm
from helper_tf_glm import get_impute_reusable
from helper_tf_irls_ig import get_reusable
#%%
#import helper_dag
global count
count = 0
import helper_erm_nn 
import helper_load_nn

#import helper_erm_nn_stateless

debug_loss = 0
 
def _get_loss(self, inner_model, x_in):
    
    idx_target = np.array([0])
    
    if self.loss_type == 0:
        X_target = self.X_target
        y_target = self.y_target
        y_pred_0 = inner_model(X_target)
        
        y_pred = tf.reshape(y_pred_0,-1)
    #    import ipdb;ipdb.set_trace()
        
        y_flip = 1 - y_target
    
    #    loss_0 = keras.losses.BinaryCrossentropy(from_logits=True)(y_target, y_pred)
    #    loss = -loss_0
    
        if debug_loss:
            loss = inner_model.trainable_variables[0][0]**2
        else:
            loss = keras.losses.BinaryCrossentropy(from_logits=True)(y_flip, y_pred)        
        
    elif self.loss_type == 1:        
        
        with tf.GradientTape() as tape:            
            tape.watch(x_in)
            y_est = inner_model(x_in, training=False)
            
        grads = tf.stack(tape.gradient(y_est, x_in))
        grads_target = tf.gather(grads, idx_target,axis=1)
        
#        target_feature_score = tf.gather(p_r_x,np.array([0]),axis=1)*tf.abs(grads_target)
#        target_feature_score = tf.reduce_mean(target_feature_score,0)
        target_feature_score = tf.reduce_mean(tf.abs(grads_target),0)
        
        if target_feature_score.ndim != 1:
            loss = tf.reduce_sum(target_feature_score)
        else:
            loss = tf.squeeze(target_feature_score)
    elif self.loss_type == 2:
#        import ipdb;ipdb.set_trace()
        weights = tf.gather(inner_model.layers[0].weights[0], idx_target,axis=0)
#        loss = tf.reduce_mean(tf.abs(weights))
        loss = tf.reduce_mean(weights**2)
        
    elif self.loss_type == 3:
        
        logits_adv = self.model_adv(x_in)        
        logits = inner_model(x_in, training=False)
        
#        loss = tf.reduce_mean(logits_adv - logits)**2        
        prob_adv = tf.math.sigmoid(logits_adv)
        loss = keras.losses.BinaryCrossentropy(from_logits=True)(prob_adv, logits)
            
#        soft_teacher = tf.math.sigmoid(logits_teacher)
                
#        loss_kd = binary_cross_entropy_with_logits(logits_student/self.T,
#                                                           soft_teacher)
                
    elif self.loss_type in [4,5,6]:
#        import ipdb;ipdb.set_trace()
        X = tf.gather(x_in,np.arange(x_in.shape[1]-1),axis=1)
#        tf.stack([X,tf.on])
        
        X_0 = tf.concat([X,tf.zeros([len(X),1])],1)
        X_1 = tf.concat([X,tf.ones([len(X),1])],1)        
        
        if self.family == 'reg':
            est_0 = inner_model(X_0)
            est_1 = inner_model(X_1)
        else:
            est_0 = tf.math.sigmoid(inner_model(X_0))
            est_1 = tf.math.sigmoid(inner_model(X_1))
        
        ate = tf.reduce_mean(est_1-est_0)
        if self.loss_type == 4:
            loss = ate**2
        elif self.loss_type == 5:
            loss = tf.abs(ate)
        elif self.loss_type == 6:
            true_ate = .1
            loss = tf.abs(ate-true_ate)
        print('ate',ate.numpy())
        print('bias %.2f'%np.log10(np.abs(inner_model.get_weights()[-1][0])))
#        if ate.numpy() == 0:
#            import ipdb;ipdb.set_trace()

        
#    loss_0 = inner_model.compiled_loss(y_target, y_pred)
#    compute_loss(y=y_target, y_pred=y_pred)    
    
    
    return loss
    
def get_loss(self, inner_model, 
             z_sub,
             x_in, y,
             return_all = False):
    
    '''
    Computes \ell(\phi,\theta)
    '''
    
    bool_full = 1
    p_r_x = self(z_sub, training=True)  # Forward pass                                    
    
    loss = _get_loss(self, inner_model, x_in)
    #%%                                    
    plain_loss = loss.numpy()
    
    if bool_full:
#        prob_obs = helper_em_tf.get_obs_prob(p_r_x, len(self.idx_mask))
        prob_obs = helper_load_nn.get_obs_prob_wrap(p_r_x, len(self.idx_mask),
                                                    self.n_out)
        
        if self.reg_lmbda != 0:
            loss = loss + self.reg_lmbda*(1-prob_obs)                                
    
    if not return_all:
        return loss
    else:
        return loss, plain_loss
                            
def get_f_y_fyy(self, inner_model, z_sub, 
                x_in, y,
                idx_mask_in):
    
    '''
    Computes the gradient and hessian of the f_tilde w.r.t. \theta
    '''
#    import ipdb;ipdb.set_trace()
    
    p_r_x = self(z_sub, training=True)  # Forward pass                                    
    
    if self.type_modeler != 'cca':
        kwargs_imp = get_impute_reusable(x_in, y,
                                         p_r_x,
                                         idx_adv = idx_mask_in,
                                         type_impute = self.type_impute,
                                         bool_o_mat = self.bool_o_mat,
                                         bool_bias = True,
                                         family = None)
    else:
        kwargs_imp = {}
    
    f_y, f_yy = helper_erm_nn._get_f_y_fyy(x_in, y,
                             inner_model,
                             p_r_x, 
                             idx_adv = idx_mask_in,
                             type_impute = self.type_impute,
                             bool_cca = self.type_modeler == 'cca',
                             kwargs_imp = kwargs_imp,
                             family = self.family)
        
    return f_y, f_yy

def get_grad(self, z,
             return_loss = False):
    
    import ipdb;ipdb.set_trace()
    z_sub, x_in, \
    y, idx_mask_in = get_reusable(self, z)
    
    if 0:
        inner_model, metrics_inner = helper_erm_nn.solve_y(self, z_sub, 
                                                           x_in, y,
                                                           idx_mask_in)
        
        if 'solver_warm_start' in self.kwargs_solver:
            if self.kwargs_solver['solver_warm_start']:
                print('warm_starting')
                self.prev_solution = inner_model
                
    else:
        self.family = 'lr'
        self.glm_lmbda = 0
        self.n_steps = 10
        self.max_steps = 1000
        
        import helper_tf_irls_ig
        beta = helper_tf_irls_ig.solve_y(self, z_sub, 
                                         x_in, y,
                                         idx_mask_in)
        
#        import ipdb;ipdb.set_trace()
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.initializers import Constant
        print('using new inner')
        inner_model = Sequential([
                            Dense(
                                1,
                                input_shape=(2,),     # Input shape
                                kernel_initializer=Constant(tf.expand_dims(beta,1)),
                                activation=None,use_bias=False
                            )
                        ])
#        import ipdb;ipdb.set_trace()
#        inner_model.trainable_variables[0] = beta
    
    #Calculates the gradients of the outer objective
    with tf.GradientTape() as tape:        
        tape.watch(inner_model.trainable_variables)
                                                            
        #Calculates the loss
        loss, plain_loss = get_loss(self, inner_model, z_sub, 
                                    x_in, return_all = 1,
                                    y = y)
#        import ipdb;ipdb.set_trace()
        
        all_vars = self.trainable_variables + inner_model.trainable_variables
        grad_all = tape.gradient(loss, all_vars)
        
    grad_x_1_l = grad_all[:len(self.trainable_variables)]
    
    grad_beta_0 = grad_all[len(self.trainable_variables):]
    for i in range(len(grad_beta_0)):
        if grad_beta_0[i] is None:
            grad_beta_0[i] = tf.zeros_like(inner_model.trainable_variables[i])
        
    grad_beta = helper_erm_nn.get_flat(grad_beta_0)
#    grad_beta = helper_erm_nn.get_flat(grad_all[len(self.trainable_variables):])
    
    #Calculates the derivatives of the inner objective
    with tf.GradientTape() as tape:        
        _g, _H = get_f_y_fyy(self, inner_model, 
                             z_sub, x_in, 
                             y, idx_mask_in)
        
        _g_norm = np.linalg.norm(tf.squeeze(_g),1)
#        print('_g',_g_norm)
    debug_glm = 0
    if debug_glm:
        import helper_tf_irls_ig
        
        self.family = 'lr'
        self.glm_lmbda = 0
        a,b = helper_tf_irls_ig.get_f_y_fyy(self, 
                                            tf.squeeze(inner_model.trainable_variables[0]), 
                                            z_sub, x_in, y,
                                            idx_mask_in)
#    import ipdb;ipdb.set_trace()
    #Calculates f_yx
    f_x_l = tape.jacobian(_g, self.trainable_variables)
    
    if "solver_arch" not in self.kwargs_solver.keys():
#        == 'mlp':
        bool_inv = 1
    else:
#    elif self.kwargs_solver.solver_arch == 'rnn':
        bool_inv = 0
        
#    import ipdb;ipdb.set_trace()
    if bool_inv:
#        H_inv = np.linalg.inv(_H.numpy())
#        H_inv = tf.constant(H_inv)
        H_inv = tf.linalg.inv(_H)
        A1 = tf.einsum('ij,j->i', H_inv, grad_beta)
        grad_x_2_l = []
        
        for f_x in f_x_l:
#            import ipdb;ipdb.set_trace()
            shp = f_x.shape[1:]
            temp = tf.reshape(f_x,(f_x.shape[0],-1))
            temp2 = tf.einsum('ij,i->j', temp, A1)
            A2 = tf.reshape(temp2, shp)
            A2 = -A2
            grad_x_2_l.append(A2)
    else:
        #A more efficient lstsq can be possible
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
#    grad = [grad_x_1+grad_x_2 for grad_x_1,grad_x_2 in temp_iter] 
    
    grad = []
    for grad_x_1, grad_x_2 in temp_iter:
        
        if grad_x_1 is not None:
            print('norm_ratio',
                  np.linalg.norm(grad_x_1.numpy().ravel(),1)\
                  /np.linalg.norm(grad_x_2.numpy().ravel(),1))
            grad.append(grad_x_1+grad_x_2)
        else:
            grad.append(grad_x_2)
        
            
    if not return_loss:
        return grad
    else:
        p_r_x = self(z_sub, training=True)  # Forward pass
        
        if 0:
            temp_check = self(z_sub, training=False)
            if np.any(temp_check.numpy() != p_r_x.numpy()):
                print('dropout can cause a problem since p_r_x called multiple')
        
        metrics = dict(loss=loss, 
                       plain_loss=plain_loss,
                       _g_norm=_g_norm)
        
        metrics.update(metrics_inner)
        
        return grad, p_r_x, \
                inner_model, metrics
                
def get_grad_unroll(self, z,
                    return_loss = False):
    
#    import ipdb;ipdb.set_trace()
    z_sub, x_in, \
    y, idx_mask_in = get_reusable(self, z)
            
#        if 'solver_warm_start' in self.kwargs_solver:
#            if self.kwargs_solver['solver_warm_start']:
#                print('warm_starting')
#                self.prev_solution = inner_model                
    
    #Calculates the gradients of the outer objective
    with tf.GradientTape() as tape:
#        tape.watch()
        
        inner_model, metrics_inner = helper_erm_nn_stateless.solve_y(self, 
                                                                     z_sub, 
                                                                     x_in, y,
                                                                     idx_mask_in)
                        
        #Calculates the loss
        loss, plain_loss = get_loss(self, inner_model, z_sub, 
                                    x_in, return_all = 1,
                                    y = y)
#        import ipdb;ipdb.set_trace()
        grad = tape.gradient(loss, self.trainable_variables)
        
#        print('_g',_g_norm)    
#        print('bias',self.trainable_variables[-1].numpy())
#        print('_g',np.sum([np.abs(grad_i.numpy()).sum() for grad_i in grad]))
#    import ipdb;ipdb.set_trace()
    #Calculates f_yx
    
    if 'solver_warm_start' in self.kwargs_solver:
        if self.kwargs_solver['solver_warm_start']:
            print('warm_starting')
            self.prev_solution = inner_model
                
    if not return_loss:
        return grad
    else:
        p_r_x = self(z_sub, training=True)  # Forward pass
        
        if not hasattr(self,'bool_fixed'):
            temp_check = self(z_sub, training=False)
            if np.any(temp_check.numpy() != p_r_x.numpy()):
                print('dropout can cause a problem since p_r_x called multiple')
        
        metrics = dict(loss=loss, 
                       plain_loss=plain_loss,
#                       _g_norm=_g_norm
                       )
        
        metrics.update(metrics_inner)
        
        return grad, p_r_x, \
                inner_model, metrics
                
def _get_A_mvp(H_mvp, alpha, gamma):
    
    return alpha - gamma*H_mvp

def _get_B_mvp(J_mvp_l, gamma):
    
    B_mvp_l = []
    
    for J_mvp_i in J_mvp_l:
        if J_mvp_i is None:
            print('J_mvp_i is None')
        else:
            B_mvp_l.append(-gamma*J_mvp_i)
            
    return B_mvp_l

def get_A_B_mvp(self, inner_model, z_sub, 
                x_in, y,
                idx_mask_in, alpha,
                gamma,
                verbose = 0):
        
    debug = False
    params_outer = self.trainable_variables
    with tf.GradientTape(persistent=debug) as tape:        
        
        _g, H_mvp = get_f_y_f_yy_mvp(self, inner_model, 
                                  z_sub, x_in, 
                                  y, idx_mask_in,
                                  alpha)
        
        if debug:
            f_y, f_yy = get_f_y_fyy(self, inner_model, z_sub, 
                        x_in, y,
                        idx_mask_in)
        
    
    if verbose: print('_g',np.linalg.norm(tf.squeeze(_g),1))
    A_mvp = _get_A_mvp(H_mvp, alpha, gamma)
    
    J_mvp_l = tape.gradient(_g, params_outer, output_gradients=alpha)
    
    if debug:
        J_l = tape.jacobian(_g, params_outer)
        for J_i,J_mvp_i in zip(J_l,J_mvp_l):
    #        temp = tf.einsum('ijk,i->jk',J_i,alpha)
            if J_i.ndim == 3:
                print(np.allclose(tf.einsum('ijk,i->jk',J_i,alpha),J_mvp_i,atol=1e-7)
                      )
            else:
                print(np.allclose(tf.einsum('ij,i->j',J_i,alpha),J_mvp_i))    
    
    B_mvp_l = _get_B_mvp(J_mvp_l, gamma)
    
    return A_mvp, B_mvp_l
        
def get_f_y_f_yy_mvp(self, inner_model, z_sub, 
                     x_in, y,
                     idx_mask_in, alpha):
    
    '''
    Computes the gradient and hessian of the f_tilde w.r.t. \theta
    '''
#    import ipdb;ipdb.set_trace()
    
    p_r_x = self(z_sub, training=True)  # Forward pass                                    
    
    if self.type_modeler != 'cca':        
        kwargs_imp = get_impute_reusable(x_in, y,
                                     p_r_x,
                                     idx_adv = idx_mask_in,
                                     type_impute = self.type_impute,
                                     bool_o_mat = self.bool_o_mat,
                                     bool_bias = True,
                                     family = None)
    else:            
        kwargs_imp = {}
           
    g, H_mvp = helper_erm_nn._get_fyy_mvp(x_in, y,
                                         inner_model,
                                         p_r_x, 
                                         idx_adv = idx_mask_in,
                                         type_impute = self.type_impute,
                                         alpha = alpha,
                                         bool_cca = self.type_modeler == 'cca',
                                         kwargs_imp = kwargs_imp,
                                         family = self.family)
     
    return g, H_mvp
 
def _get_grad_rmd(self, 
                  inner_model_l, 
                  gamma, z_sub, 
                  x_in, y,
                  idx_mask_in,
                  grad_x_1_l, 
                  grad_beta):
    
    K = len(inner_model_l)
    alpha = grad_beta
    h = grad_x_1_l
#    import ipdb;ipdb.set_trace()    
    
    for i in range(K-1):
        
        inner_model = inner_model_l[-2-(i)]
        A_mvp, B_mvp_l = get_A_B_mvp(self, inner_model, z_sub, 
                                   x_in, y,
                                   idx_mask_in, alpha,
                                   gamma)
                
        for i in range(len(h)):        
            if h[i] is not None:
                h[i] = h[i] + B_mvp_l[i]
            else:
                print('grad missing')
                h[i] = B_mvp_l[i]
                
#        h = h + B_mvp
        alpha = A_mvp
    
    return h

def get_grad_rmd(self, z,
                 return_loss = False):
    
#    import ipdb;ipdb.set_trace()
    z_sub, x_in, \
    y, idx_mask_in = get_reusable(self, z)
        
    inner_model_l, metrics_inner = helper_erm_nn.solve_y(self, z_sub, 
                                                         x_in, y,
                                                         idx_mask_in,
                                     n_return_states = self.n_return_states)
        
#    if 'solver_warm_start' in self.kwargs_solver:
#        raise ValueError
    if self.kwargs_solver['solver_warm_start']:
        print('warm_starting')
        self.prev_solution = inner_model_l[-1]
        
    #Calculates the gradients of the outer objective
    
    with tf.GradientTape() as tape:        
        tape.watch(inner_model_l[-1].trainable_variables)
                                                            
        #Calculates the loss
        loss, plain_loss = get_loss(self, inner_model_l[-1], z_sub, 
                                    x_in, return_all = 1,
                                    y = y)
#        import ipdb;ipdb.set_trace()
        all_vars = self.trainable_variables + \
                    inner_model_l[-1].trainable_variables
        grad_all = tape.gradient(loss, all_vars)
        
    grad_x_1_l = grad_all[:len(self.trainable_variables)]
    
    grad_beta_0 = grad_all[len(self.trainable_variables):]
    for i in range(len(grad_beta_0)):
        if grad_beta_0[i] is None:
            grad_beta_0[i] = tf.zeros_like(inner_model_l[-1].trainable_variables[i])
        
    grad_beta = helper_erm_nn.get_flat(grad_beta_0)
#    grad_beta = helper_erm_nn.get_flat(grad_all[len(self.trainable_variables):])
    
    gamma = self.kwargs_solver['solver_lr']
    grad = _get_grad_rmd(self, 
                  inner_model_l, 
                  gamma, z_sub, 
                  x_in, y,
                  idx_mask_in,
                  grad_x_1_l, 
                  grad_beta)    
    
        
    if not return_loss:
        return grad
    else:
        p_r_x = self(z_sub, training=True)  # Forward pass
        
        temp_check = self(z_sub, training=False)
        if np.any(temp_check.numpy() != p_r_x.numpy()):
            print('dropout can cause a problem since p_r_x called multiple')
        
        metrics = dict(loss=loss, 
                       plain_loss=plain_loss,
#                       _g_norm=_g_norm
                       )
        
        metrics.update(metrics_inner)
        
        return grad, p_r_x, \
                inner_model_l[-1], metrics

def get_grad_pen(self, z,
                 return_loss = False):
    
    eta = 0.5
        
    z_sub, x_in, \
    y, idx_mask_in = get_reusable(self, z)
#        
#    import ipdb;ipdb.set_trace()
#    [for self.tra]
    if self.prev_solution is None:
        
        temp_kwargs = copy.deepcopy(self.kwargs_solver)
        temp_kwargs['solver_epochs'] = 300
        temp_kwargs['solver_optimizer'] = 'adam'
        temp_kwargs['solver_lr'] = 1e-1
        
        if 'solver_optimizer_kwargs' in temp_kwargs.keys():
            temp_kwargs.pop('solver_optimizer_kwargs')        
        
        temp_p_r_x = np.stack([np.ones(len(x_in)),np.zeros(len(x_in))],1).astype(np.float32)
        inner_model, _ = helper_erm_nn._solve_y(x_in, y, temp_p_r_x,
                                             idx_mask_in,
                                             self.type_impute, 
                                             self.type_modeler,
                                             kwargs_imp = {},
                                             **temp_kwargs)
            
        self.prev_solution = inner_model
        
        if self.kwargs_solver['solver_optimizer'] == 'adam':
            self.optimizer_lower = keras.optimizers.Adam(learning_rate=self.kwargs_solver['solver_lr'])
        elif self.kwargs_solver['solver_optimizer'] == 'sgd':
            self.optimizer_lower = keras.optimizers.SGD(learning_rate=self.kwargs_solver['solver_lr'])
            
    else:
        inner_model = self.prev_solution
    
    inner_model_star, metrics_inner = helper_erm_nn.solve_y(self, z_sub, 
                                                            x_in, y,
                                                            idx_mask_in)                

    all_vars = self.trainable_variables + inner_model.trainable_variables
                
    #Calculates the gradients of the outer objective
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:  
            
            tape1.watch(inner_model.trainable_variables)
            tape2.watch(inner_model.trainable_variables)
            #Calculates the loss
            upper_obj, plain_loss = get_loss(self, inner_model, z_sub, 
                                             x_in, return_all = 1,
                                             y = y)
            
            lower_obj = get_f_tilde_wrap(self, inner_model, z_sub, 
                                         x_in, y,
                                         idx_mask_in)
            
            lower_obj_star = get_f_tilde_wrap(self, inner_model_star, z_sub, 
                                              x_in, y,
                                              idx_mask_in)
            
            q = lower_obj - lower_obj_star
    
#    import ipdb;ipdb.set_trace()
    
    grad_loss = tape1.gradient(upper_obj, all_vars)            
    grad_gap = tape2.gradient(q, all_vars)    
    
    #Computes the weight of the gap     
    inp = get_inp(grad_loss, grad_gap)
    q_norm = get_inp(grad_gap, grad_gap)    
    phi = eta*q_norm    
    lmbda = np.max([(phi - inp)/q_norm,0])
    
#    import ipdb;ipdb.set_trace()
#    lmbda = 0
#    lmbda = 1
#    lmbda = 100
    print('lmdda %.2f, inp %.2f'%(lmbda,inp/q_norm))
    
    #Computes the total gradient    
    grad_all = get_add(grad_loss, grad_gap, lmbda)
    
    grad_upper = grad_all[:len(self.trainable_variables)]
    grad_lower = grad_all[len(self.trainable_variables):]
    
    norm_upper = [np.linalg.norm(grad_u.numpy().ravel(),1) for grad_u in grad_upper]
    norm_lower = [np.linalg.norm(grad_l.numpy().ravel(),1) for grad_l in grad_lower]
    print('grad ratio', np.sum(norm_upper)/np.sum(norm_lower))
#    import ipdb;ipdb.set_trace()
    self.optimizer_lower.apply_gradients(zip(grad_lower, 
                                             inner_model.trainable_variables))
    self.prev_solution = inner_model
    
    grad = grad_upper
    loss = upper_obj
    
    penalty_obj = upper_obj + lmbda*q
    
    if not return_loss:
        return grad
    else:
        p_r_x = self(z_sub, training=True)  # Forward pass
        
        temp_check = self(z_sub, training=False)
        if np.any(temp_check.numpy() != p_r_x.numpy()):
            print('dropout can cause a problem since p_r_x called multiple')
        
        metrics = dict(loss=loss, 
                       plain_loss=plain_loss,
                       penalty_obj=penalty_obj,
                       q=q,
#                       _g_norm=_g_norm
                       )
        
        metrics.update(metrics_inner)
        
        return grad, p_r_x, \
                inner_model, metrics
#    import ipdb;ipdb.set_trace()    
    
def get_f_tilde_wrap(self, inner_model, z_sub, 
                     x_in, y,
                     idx_mask_in):
    
    '''
    Computes the gradient and hessian of the f_tilde w.r.t. \theta
    '''
#    import ipdb;ipdb.set_trace()
    
    p_r_x = self(z_sub, training=True)  # Forward pass                                    
    
    if self.type_modeler != 'cca':                
        kwargs_imp = get_impute_reusable(x_in, y,
                                         p_r_x,
                                         idx_adv = idx_mask_in,
                                         type_impute = self.type_impute,
                                         bool_o_mat = self.bool_o_mat,
                                         bool_bias = True,
                                         family = None)    
    else:
        kwargs_imp = {}
    
    f_tilde = helper_erm_nn.get_f_tilde(x_in, y, inner_model, 
                                        p_r_x,
                                        idx_adv = idx_mask_in,
                                        type_impute = self.type_impute, 
                                        bool_cca = self.type_modeler == 'cca',
                                        kwargs_imp = kwargs_imp,
                                        family = self.family,
                                        type_scale = self.type_scale)
    
    f_tilde = -f_tilde
    
    return f_tilde

def get_inp(grad_1_l, grad_2_l):
    
    inp = 0
    for grad_1, grad_2 in zip(grad_1_l,grad_2_l):
        if not (grad_1 is None or grad_2 is None):
            inp = inp + tf.math.reduce_sum(grad_1 * grad_2)

    return inp

def get_add(grad_1_l, grad_2_l,
            lmbda):
    
    grad = []
    for grad_x_1, grad_x_2 in zip(grad_1_l, grad_2_l):
        
        if grad_x_1 is not None:
#            print('norm_ratio',
#                  np.linalg.norm(grad_x_1.numpy().ravel(),1)\
#                  /np.linalg.norm(grad_x_2.numpy().ravel(),1))
            grad.append(grad_x_1+lmbda*grad_x_2)
        else:
            grad.append(lmbda*grad_x_2)
            
    return grad

def get_grad_inv(self, z,
                 return_loss = False):
    
#    import ipdb;ipdb.set_trace()
    z_sub, x_in, \
    y, idx_mask_in = get_reusable(self, z)
    
    if 1:
        inner_model, metrics_inner = helper_erm_nn.solve_y(self, z_sub, 
                                                           x_in, y,
                                                           idx_mask_in)
        
        if 'solver_warm_start' in self.kwargs_solver:
            if self.kwargs_solver['solver_warm_start']:
                print('warm_starting')
                self.prev_solution = inner_model                
    
    #Calculates the gradients of the outer objective
    with tf.GradientTape() as tape:
        tape.watch(inner_model.trainable_variables)
                                                            
        #Calculates the loss
        loss, plain_loss = get_loss(self, inner_model, z_sub, 
                                    x_in, return_all = 1,
                                    y = y)
#        import ipdb;ipdb.set_trace()

        all_vars = self.trainable_variables + inner_model.trainable_variables
        grad_all = tape.gradient(loss, all_vars)
        
    grad_x_1_l = grad_all[:len(self.trainable_variables)]
    
    grad_beta_0 = grad_all[len(self.trainable_variables):]
    for i in range(len(grad_beta_0)):
        if grad_beta_0[i] is None:
            grad_beta_0[i] = tf.zeros_like(inner_model.trainable_variables[i])
        
    grad_beta = helper_erm_nn.get_flat(grad_beta_0)
    
    #Calculates the derivatives of the inner objective
    with tf.GradientTape() as tape:        
        _g, _H = get_f_y_fyy(self, inner_model, 
                             z_sub, x_in, 
                             y, idx_mask_in)
        
        _g_norm = np.linalg.norm(tf.squeeze(_g),1)

    v = np.linalg.lstsq(_H.numpy(),grad_beta.numpy())[0]  #n_beta,
    v = -v
    
    grad_x_2_l = tape.gradient(_g, self.trainable_variables, 
                               output_gradients=v)
    
    temp_iter = zip(grad_x_1_l,grad_x_2_l)
#    grad = [grad_x_1+grad_x_2 for grad_x_1,grad_x_2 in temp_iter] 
    
    grad = []
    for grad_x_1, grad_x_2 in temp_iter:
        
        if grad_x_1 is not None:
            print('norm_ratio',
                  np.linalg.norm(grad_x_1.numpy().ravel(),1)\
                  /np.linalg.norm(grad_x_2.numpy().ravel(),1))
            grad.append(grad_x_1+grad_x_2)
        else:
            grad.append(grad_x_2)
        
            
    if not return_loss:
        return grad
    else:
        p_r_x = self(z_sub, training=True)  # Forward pass
        
        if not hasattr(self, 'bool_fixed'):
            temp_check = self(z_sub, training=False)
            if np.any(temp_check.numpy() != p_r_x.numpy()):
                print('dropout can cause a problem since p_r_x called multiple')
        
        metrics = dict(loss=loss, 
                       plain_loss=plain_loss,
                       _g_norm=_g_norm)
        
        metrics.update(metrics_inner)
        
        return grad, p_r_x, \
                inner_model, metrics