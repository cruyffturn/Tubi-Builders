# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from helper_em_tf import get_outer_sum, gather_x2, powerset
import helper_impute_tf

from tensorflow.keras.metrics import binary_crossentropy
from sklearn.metrics import roc_auc_score

from helper_prob.models.helper_mvn_tf import get_KL_uni
from helper_scale import get_scaled
debug_imp = 0
#global count
#count = co0

def _update_total(X_o, idx_m, idx_o, 
                 mu_est_m,
                 weights = None):
    
    '''
    In:
        X_o:        N,|o|
        mu_est_m:   |m|,
        
        total_inr:  p,p
        total_x:    p,
        weights:    N,
    Inter:
        mu
        
    Out:
        x:          p,N
        inr:        p,p
    '''
    
    if len(idx_m) != 0:
#        mu_1_2, S_1_2 = get_cond_prob(X_o, idx_m, idx_o, 
#                                      mu, S)
        
#        mu_est_m = tf.gather(mu_est, indices=idx_m) # |m|    
        
        mu_1_2 = tf.repeat(tf.expand_dims(mu_est_m,0),
                           [X_o.shape[0]], 0) # N,|m|
            
        out_o = get_outer_sum(X_o, weights = weights)              #|o|,|o| +
        out_cross = get_outer_sum(X_o, mu_1_2, weights = weights)  #|o|,|m| +
        
        out_mu = get_outer_sum(mu_1_2, weights = weights)          #|m|,|m| +
        
        A = tf.concat([out_o,out_cross],1)
        
        B1 = tf.transpose(out_cross)        
        B2 = out_mu
            
        B = tf.concat([B1,B2],1)
        
    #    import ipdb;ipdb.set_trace()
        inr_unsort = tf.concat([A,B],0)
        
        if weights is None:
            C1 = X_o
            C2 = mu_1_2
        else:
            C1 = tf.transpose(X_o)*weights   #+
            C2 = tf.transpose(mu_1_2)*weights    #+
            
        x_unsort = tf.concat([C1,
                              C2],0)
    
        idx = np.argsort(np.hstack([idx_o,idx_m]))
        
        inr = gather_x2(inr_unsort, idx, idx)    
        x = tf.gather(x_unsort, idx)
    else:
        out_o = get_outer_sum(X_o, weights = weights)              #|o|,|o| +
        inr_unsort = out_o 
        
        if weights is None:
            x_unsort = tf.transpose(X_o)
        else:                    
            x_unsort = tf.transpose(X_o)*weights     #+
                              
        idx = np.argsort(np.hstack([idx_o,idx_m]))
        
        inr = gather_x2(inr_unsort, idx, idx)
        x = tf.gather(x_unsort, idx)
        
    return inr, x

def sub_glm_enum(X, 
                p_r_x, 
                idx_adv,
                bool_o_mat,
                debug =0, 
                bool_full = True):
    
    '''
    +
    In:
        X:          N,p
        idx_adv:    p_sub,        #masked variables
    Inter:
        total_inr:  #
        total_x:    #
    '''    
#    import ipdb;ipdb.set_trace()
    N,p = X.shape
    
    X_mask = tf.gather(X, indices=idx_adv, axis = 1)
    
    mu_est_0 = helper_impute_tf.get_mu(X_mask, p_r_x, 
                                       bool_o_mat)    
    
    print('mean_est',np.round(mu_est_0.numpy(),2))
#    N = X.shape[0:1]
#    p = X.shape[1]
        
    total_inr = tf.constant(tf.zeros((p,p),tf.float32))
    total_x = tf.constant(tf.zeros((p,N)),tf.float32)    
    
    if debug:
        p_r_x = tf.Variable(initial_value= tf.ones((N,p))/2, 
                          dtype='float32')    
    
#    idx_n_adv = np.setdiff1d(idx_adv, idx_m)
#            
#    N_used = 0
    p_sub = len(idx_adv)
    if not bool_full:            
        pass
        
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        if not bool_full:
            pass
            
        else:
#            import ipdb;ipdb.set_trace()
            num = tf.gather(p_r_x, indices=np.array([i]), axis = 1)
            
            if p_sub == p:
                p_miss = tf.gather(p_r_x, indices=np.array([p_r_x.shape[1]-1]), 
                                   axis = 1)
            else:
                p_miss = 0
            
            weights = tf.squeeze(num/(1-p_miss))
        
#        idx_all_o = idx_adv[idx_o] + 
        
        mu_est_m = tf.gather(mu_est_0, idx_sub_m)
        
        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
        
        if len(idx_o) != 0:

#            import ipdb;ipdb.set_trace()
            X_o = tf.gather(X, indices=idx_o, axis = 1)
            
            inr, x = _update_total(X_o, idx_m, idx_o, 
                                   mu_est_m,
                                   weights = weights
                                   )
            if debug:
                print(idx_o)
                print(inr.numpy()/N)
                print(x.numpy()/N)
#                import ipdb;ipdb.set_trace()
            total_inr = total_inr + inr
            total_x = total_x + x
#            N_used += len(X_o)                
        
    return total_x, total_inr

def glm(X, y,
        p_r_x, 
        idx_adv,
        bool_o_mat,
        debug =0, 
        bool_full = True):
    
    total_x, total_inr = sub_glm_enum(X, p_r_x, 
                                    idx_adv,
                                    bool_o_mat,
                                    debug = debug, 
                                    bool_full = bool_full)
    
#    import ipdb;ipdb.set_trace()
    
    A = total_inr
    b = total_x @ tf.expand_dims(y,1)
    
    beta = tf.linalg.inv( A) @ b
    
    return beta

def sub_irls_enum(X, y,
                  beta,
                  p_r_x, 
                  idx_adv,
                  bool_o_mat,
                  family,
                  bool_bias,
                  type_impute,
                  debug = 0, 
                  bool_full = True,
                  verbose = 0,
                  bool_warn_inv = False,
#                  bool_warn_inv = 1,
                  bool_cca = False,
                  lmbda = 0,
                  return_grad = False,
                  bool_solve_np = False,
                  kwargs_imp = None,
                  bool_tf_inv = False,
                  type_scaler = None):
    
    '''
    Modifies Algorithm 8.1 in \cite{Murphy}
    
    +
    In:
        X:          N,p
        idx_adv:    p_sub,        #masked variables
    Inter:
        total_inr:  #
        total_x:    #
    Out:
        direct:     #direction used in the irls
            
    '''    
    
#    import ipdb;ipdb.set_trace()
    N,p = X.shape
    
    if kwargs_imp is None:
        if len(idx_adv) != 0:
            kwargs_imp = get_impute_reusable(X, y,
                                             p_r_x,
                                             idx_adv,
                                             type_impute,
                                             bool_o_mat,
                                             bool_bias,
                                             family,
                                             verbose=verbose
                                             )
        else:
            kwargs_imp = {}
        
        
    total_g = tf.constant(tf.zeros((p,)),tf.float32)
    total_H = tf.constant(tf.zeros((p,p),tf.float32))        
    
    if debug:
        p_r_x = tf.Variable(initial_value= tf.ones((N,p))/2, 
                          dtype='float32')    
    
#    idx_n_adv = np.setdiff1d(idx_adv, idx_m)
#            
#    N_used = 0
    p_sub = len(idx_adv)
    if not bool_full:            
        pass
        
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        if not bool_full:
            pass
            
        else:
            num = tf.gather(p_r_x, indices=np.array([i]), axis = 1)
            
            if ((p_sub == p) and not bool_bias) or \
               ((p_sub == p-1) and bool_bias):
#                if bool_bias: print('handling the bias term')
#                print('Possible error handling the bias term.')
                p_miss = tf.gather(p_r_x, indices=np.array([p_r_x.shape[1]-1]), 
                                   axis = 1)                
            else:
                p_miss = 0
            
            weights = tf.squeeze(num/(1-p_miss))                
        
        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
                
        if ((len(idx_o) != 0) and not bool_bias) or \
           ((len(idx_o) != 1) and bool_bias):
               
           #Checks at least one original feature is observed
               
#            if bool_bias: print(len(idx_o),'handling the bias term')

#            import ipdb;ipdb.set_trace()
            if debug_imp:
                X_o = tf.gather(X, indices=idx_o, axis = 1)
                mu_est_m = tf.gather(kwargs_imp['mu_est_0'], idx_sub_m)
            
            K = 2
            for k in range(K):
                X_hat_0, prob = get_impute(X, y, idx_o, idx_m,
                                   idx_sub_m = idx_sub_m,
                                   type_impute = type_impute,
                                   k=k,
                                   **kwargs_imp)
                
                if type_scaler is not None:
                    X_hat = get_scaled(X_hat_0, type_scaler)
                else:
                    X_hat = X_hat_0
                     
                if prob is not None:
                    weights_0 = weights * prob
                else:
                    weights_0 = weights
                                        
                eta_hat = tf.einsum('ij,j->i', X_hat, beta) 
                if debug_imp:
                    eta_hat2 = _get_eta(X_o, beta,
                                       idx_m, idx_o, 
                                       mu_est_m)
                    if not np.allclose(eta_hat.numpy(),eta_hat2.numpy()): raise ValueError
                
                #Calculates the gradient
                a_p = get_a_p(eta_hat, family)
                
                weights_1 = weights_0 * (y - a_p)
                
                x = tf.transpose(X_hat)*weights_1   #p,N
                if debug_imp:            
                    x2 = _update_x(X_o, idx_m, idx_o, 
                                  mu_est_m,
                                  weights = weights_1)        #p,N                        
                    if not np.allclose(x.numpy(),x2.numpy()): raise ValueError
    
                g = tf.math.reduce_sum(x, -1)             #p,            
                
                if lmbda != 0:
                    g = g + lmbda*beta
                    
                #Calculates the Hessian
                a_pp = get_a_pp(eta_hat, family)
                weights_2 = weights_0 * a_pp
                
                H_0 = get_outer_sum(X_hat, weights = weights_2)   #p,p
                if debug_imp:
                    H_02 = _update_outer(X_o, idx_m, idx_o, 
                                        mu_est_m,
                                        weights = weights_2)
                    
                    if not np.allclose(H_0.numpy(),H_02.numpy(),rtol=1e-3):  
                        import ipdb;ipdb.set_trace()
                        raise ValueError
    
                H = -H_0
                
                if lmbda != 0:
                    H = H + lmbda*tf.eye(p)
                    
                if debug:
                    print(idx_o)
                    print(H.numpy()/N)
                    print(g.numpy()/N)
    #                import ipdb;ipdb.set_trace()
                total_g = total_g + g
                total_H = total_H + H
                
                if prob is None:
                    break
        
        if bool_cca:
#            print('irls cca')
            break
            
#            N_used += len(X_o)                    
    
#    direct = tf.linalg.inv(total_H) @ -tf.expand_dims(total_g,1)
#    direct = tf.squeeze(direct)
#    import ipdb;ipdb.set_trace()
#    total_H = tf.cast(total_H,tf.float64)
    if not return_grad:
        if ((lmbda == 0) and not bool_solve_np) or bool_tf_inv:
            try:
                H_inv = tf.linalg.inv(total_H)
    #            H_inv = tf.cast(H_inv,tf.float32)
            except:
                import ipdb;ipdb.set_trace()
                if bool_warn_inv:
    #                import ipdb;ipdb.set_trace()
                    eps = 1e-1
                    print('Not invertible')
                    H_inv = tf.linalg.inv(total_H+eps*tf.eye(p))
                else:
                    raise ValueError
                
        #    direct = tf.einsum('ij,j->i', tf.linalg.inv(total_H), -total_g)
            direct = tf.einsum('ij,j->i', H_inv, -total_g)
        else:
            if not bool_solve_np:
        #        import ipdb;ipdb.set_trace()
                print('using Ax=b')
                direct_0 = tf.linalg.lstsq(total_H,-tf.expand_dims(total_g,1),
        #                                   l2_regularizer=1e-1
#                                           fast=False
                                           )
            else:
#                print('using numpy solve')
                direct_0 = np.linalg.lstsq(total_H,-total_g)[0]
                direct_0 = tf.constant(direct_0)
                
            direct = tf.squeeze(direct_0)
            
        return direct
    else:
        return total_g, total_H

def get_irls_enum(X, y, p_r_x, 
                  idx_adv,
                  family,
                  bool_o_mat,
                  bool_bias, 
                  type_impute,
                  beta_0 = None,
                  n_steps = 10,
                  bool_full = True,
                  bool_history = False,
                  bool_while = False,
                  max_steps = 1000,
                  bool_cca = False,
                  lmbda = 0,
                  bool_solve_np = False,
                  verbose = 0,
                  bool_tf_inv = False,
                  type_scaler = None):
    
    '''
    +
    In:
        X:      N,p
        y:      N,
        beta_0:   p,
        idx_adv:            #~Indices to the masked variables
        bool_bias:          #Used for accessing at least one feature is obs.
    '''
    if bool_cca and len(idx_adv) > 1:
        raise ValueError
        
    if beta_0 is None:
        raise ValueError
    
    if len(idx_adv) != 0:
        kwargs_imp = get_impute_reusable(X, y,
                                         p_r_x,
                                         idx_adv,
                                         type_impute,
                                         bool_o_mat,
                                         bool_bias,
                                         family,
                                         verbose=verbose
                                         )
    else:
        kwargs_imp = {}
            
    beta_prev = beta_0
    
#    log_like_prev = get_lower_bound(X, mu_prev, 
#                                    S_prev, p_r_x, 
#                                    bool_full, idx_adv)
    if family == 'normal':
        bool_while = False
        n_steps = 1
    
    if family == 'normal':
        log_like_prev = 1        #Not implemented yet
    else:    
        log_like_prev = get_log_like(X, y, beta_prev, 
                                     p_r_x,
                                     idx_adv,
                                     bool_o_mat,
                                     family, 
                                     bool_bias,
                                     type_impute,
                                     bool_full = bool_full,
                                     bool_cca = bool_cca,
                                     kwargs_imp = kwargs_imp,
                                     type_scaler = type_scaler)
            
    if bool_while:
        n_steps = max_steps
        
    for i in range(n_steps):

        if family != 'normal':
            log_like_new = get_log_like(X, y, beta_prev, 
                                         p_r_x,
                                         idx_adv,
                                         bool_o_mat,
                                         family, 
                                         bool_bias,
                                         type_impute,
                                         bool_full = bool_full,
                                         bool_cca = bool_cca,
                                         kwargs_imp = kwargs_imp,
                                         type_scaler = type_scaler)

        
#        log_like_new = get_lower_bound(X, mu_prev, 
#                                       S_prev, 
#                                       p_r_x, bool_full, idx_adv)
                    
        
        direct = sub_irls_enum(X, y,
                                  beta_prev,
                                  p_r_x, 
                                  idx_adv,
                                  bool_o_mat,
                                  family,
                                  bool_bias,
                                  bool_full = bool_full,
                                  bool_cca = bool_cca,
                                  lmbda = lmbda,
                                  bool_solve_np = bool_solve_np,
                                  type_impute = type_impute,
                                  kwargs_imp = kwargs_imp,
                                  bool_tf_inv = bool_tf_inv,
                                  type_scaler = type_scaler)
        
        beta_new = beta_prev + direct                          

        if family == 'normal':
            diff = np.inf        #Not implemented yet
            log_like_new = 1.
        else:
            diff_0 = (log_like_new.numpy() - log_like_prev.numpy())/log_like_prev.numpy()
            diff = np.abs(diff_0)*100
#        diff = 1
        
#        import ipdb;ipdb.set_trace()
        log_like_prev = log_like_new
        beta_prev = beta_new
        
        if bool_history:
            if i == 0:
                like_L = [log_like_prev]
                beta_L = [beta_prev]
            else:
                like_L.append(log_like_prev)
                beta_L.append(beta_prev)
            
#        print(i,'%.4f'%log_like_new,'%.4f'%diff)
        
        if bool_while and (diff < .00001 and i!=0):
            print('stop',i,"{:.1E}".format(diff))
            break
    
    if i == n_steps-1:
        print('not converged',i,"{:.1E}".format(diff))
            
    
    if not bool_history:            
        return beta_new
    else:
        return like_L, beta_L
    
def _get_eta(X_o, beta,
             idx_m, idx_o, 
             mu_est_m):
    
    '''
    In:
    
    Out:
        eta:    N,p
    '''
    
    mu_1_2 = tf.repeat(tf.expand_dims(mu_est_m,0),
                       [X_o.shape[0]], 0)           # N,|m|
                            
        
    x_unsort = tf.concat([X_o,
                          mu_1_2],1)
        
    idx = np.argsort(np.hstack([idx_o,idx_m]))
    
    x = tf.gather(x_unsort, idx, axis=1)
    
#    eta = tf.squeeze(x @ tf.expand_dims(beta,1))
    eta = tf.einsum('ij,j->i', x, beta)
    
    return eta
        
def _update_outer(X_o, 
               idx_m, idx_o, 
               mu_est_m,
               weights = None):
    
    '''
    In:
        X_o:        N,|o|
        mu_est_m:   |m|,
        
        weights:    N,
    Inter:
        mu
        
    Out:
        inr:        p,p
    '''
    
    if len(idx_m) != 0:        
        mu_1_2 = tf.repeat(tf.expand_dims(mu_est_m,0),
                           [X_o.shape[0]], 0) # N,|m|
            
        out_o = get_outer_sum(X_o, weights = weights)              #|o|,|o| +
        out_cross = get_outer_sum(X_o, mu_1_2, weights = weights)  #|o|,|m| +
        
        out_mu = get_outer_sum(mu_1_2, weights = weights)          #|m|,|m| +
        
        A = tf.concat([out_o,out_cross],1)
        
        B1 = tf.transpose(out_cross)
        B2 = out_mu
            
        B = tf.concat([B1,B2],1)
        
        inr_unsort = tf.concat([A,B],0)
        
    else:
        out_o = get_outer_sum(X_o, weights = weights)              #|o|,|o| +
        inr_unsort = out_o         
    
    idx = np.argsort(np.hstack([idx_o,idx_m]))        
    inr = gather_x2(inr_unsort, idx, idx)
    
    return inr

def _update_x(X_o, idx_m, idx_o, 
             mu_est_m,
             weights = None):
    
    '''
    In:
        X_o:        N,|o|
        mu_est_m:   |m|,
        
        weights:    N,
    Inter:
        mu
        
    Out:
        x:          p,N
    '''
    
    if len(idx_m) != 0:
        
        mu_1_2 = tf.repeat(tf.expand_dims(mu_est_m,0),
                           [X_o.shape[0]], 0) # N,|m|
                    
        if weights is None:
            raise ValueError
#            C1 = X_o
#            C2 = mu_1_2
        else:
            C1 = tf.transpose(X_o)*weights   #+
            C2 = tf.transpose(mu_1_2)*weights    #+
            
        x_unsort = tf.concat([C1,
                              C2],0)
    
    else:
        
        if weights is None:
            x_unsort = tf.transpose(X_o)
        else:                    
            x_unsort = tf.transpose(X_o)*weights     #+
                              
    
    idx = np.argsort(np.hstack([idx_o,idx_m]))        
    x = tf.gather(x_unsort, idx)
        
    return x

def get_a_p(eta, family):
    
    '''
    In:
        eta:    N,
    Out:
        a_p:    N,
        
    '''
    if family == 'normal':
        a_p = eta
    
    elif family == 'lr':        
        a_p = tf.math.sigmoid(eta)
        
    return a_p

        
def get_a_pp(eta, family):
    
    '''
    In:
        eta:    N,
    Out:
        a_pp:   N,
        
    '''
    
    if family == 'normal':
        a_pp = tf.ones_like(eta)
    
    elif family == 'lr':        
        sigmoid = tf.math.sigmoid(eta)    
        a_pp = (1-sigmoid)*sigmoid
        
    return a_pp

def _get_log_like(y, eta, family):
    
    a_p = get_a_p(eta, family)
    
    if family == 'lr':
        
        log_like_0 = binary_crossentropy(tf.expand_dims(y,-1), 
                                         tf.expand_dims(a_p,-1))
        log_like = -log_like_0
        
    return log_like

def get_log_like(X, y,
                beta, 
                p_r_x,
                idx_adv,
                bool_o_mat,
                family,
                bool_bias,
                type_impute,
                debug = 0, 
                bool_full = True,
                verbose = 0,
                bool_cca = False,
                kwargs_imp = None,
                type_scaler = None):
    
    '''
    In:
        X:          N,p
        idx_adv:    p_sub,        #masked variables
    '''    
    
#    import ipdb;ipdb.set_trace()
    N,p = X.shape
    
    if kwargs_imp is None:
        if len(idx_adv) != 0:
            kwargs_imp = get_impute_reusable(X, y,
                                             p_r_x,
                                             idx_adv,
                                             type_impute,
                                             bool_o_mat,
                                             bool_bias,
                                             family,
                                             verbose=verbose
                                             )
        else:
            kwargs_imp = {}
        
    if debug:
        p_r_x = tf.Variable(initial_value= tf.ones((N,p))/2, 
                          dtype='float32')    
    
    p_sub = len(idx_adv)
    if not bool_full:            
        pass
        
    total_ll = tf.constant(tf.zeros(1),tf.float32)
    
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        if not bool_full:
            pass
            
        else:
            num = tf.gather(p_r_x, indices=np.array([i]), axis = 1)
            
            if ((p_sub == p) and not bool_bias) or \
               ((p_sub == p-1) and bool_bias):

                p_miss = tf.gather(p_r_x, indices=np.array([p_r_x.shape[1]-1]), 
                                   axis = 1)                
            else:
                p_miss = 0
            
            weights = tf.squeeze(num/(1-p_miss))                
        
        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
        
        if ((len(idx_o) != 0) and not bool_bias) or \
           ((len(idx_o) != 1) and bool_bias):

#            import ipdb;ipdb.set_trace()
            if debug_imp:
               X_o = tf.gather(X, indices=idx_o, axis = 1)
               mu_est_m = tf.gather(kwargs_imp['mu_est_0'], idx_sub_m)
            
            K = 2
            for k in range(K):
                X_hat_0, prob = get_impute(X, y, idx_o, idx_m,
                                   idx_sub_m = idx_sub_m,
                                   type_impute = type_impute,
                                   k = k,
                                   **kwargs_imp)
                
                if type_scaler is not None:
                    X_hat = get_scaled(X_hat_0, type_scaler)
                else:
                    X_hat = X_hat_0
                
                if prob is not None:
                    weights_0 = weights * prob
                else:
                    weights_0 = weights
                    
                eta_hat = tf.einsum('ij,j->i', X_hat, beta)
                if debug_imp:
                    eta_hat2 = _get_eta(X_o, beta,
                                        idx_m, idx_o, 
                                        mu_est_m)
                    if not np.allclose(eta_hat.numpy(),eta_hat2.numpy()): raise ValueError
                
                ll_0 = _get_log_like(y, eta_hat, family)
                ll_1 = ll_0 * weights_0
                ll = tf.math.reduce_mean(ll_1)
            
                total_ll = total_ll + ll
                
                if prob is None:
                    break
        
        if bool_cca:
#            print('ll cca')
            break
    
    return tf.squeeze(total_ll)

def get_irls_enum_det(X, y, 
                      family,
#                      beta_0,
                      bool_intercept = True,
                      n_steps = 1,
                      bool_while = False
                      ):
    N = len(X)
    
    idx_mask = np.array([0])
    p_r_x_0 = np.array([1.,0.])
    p_r_x = np.repeat(p_r_x_0[np.newaxis,:],N,0)

    bool_o_mat = helper_impute_tf.get_obs_mat(len(idx_mask))

    if not bool_intercept:
        beta_0 = np.zeros(X.shape[1],np.float32)
        X_in = X
    else:
        beta_0 = np.zeros(X.shape[1]+1,np.float32)
        X_in = np.concatenate([np.ones((len(X),1)),X],1)
    
    import ipdb;ipdb.set_trace()
    temp_beta = get_irls_enum(X_in.astype(np.float32), 
                              y.astype(np.float32), 
                              p_r_x.astype(np.float32), 
                              idx_mask,
                              family,
                              bool_o_mat,
                              beta_0 = beta_0,
                              n_steps = n_steps,
                              bool_while = bool_while)
    
    if not bool_intercept:        
        beta = temp_beta
        c = 0
    else:
        beta = temp_beta[1:]
        c = temp_beta[0]
        
    return beta, c

def sub_irls_enum_cca(X, y,
                      beta,
                      p_r_x, 
                      idx_adv,
                      family,
                      bool_bias,
                      debug = 0, 
                      bool_full = True,
                      verbose = 0,
                      bool_warn_inv = True):
    
    '''
    Modifies Algorithm 8.1 in \cite{Murphy}
    
    +
    In:
        X:          N,p
        idx_adv:    p_sub,        #masked variables
    Inter:
        total_inr:  #
        total_x:    #
    Out:
        direct:     #direction used in the irls
            
    '''    
    
#    import ipdb;ipdb.set_trace()
    N,p = X.shape    
        

    #Loops different missing patterns
    
    num = tf.gather(p_r_x, indices=np.array([0]), axis = 1)
                        
    weights = tf.squeeze(num)          
        
#    idx_sub_m = np.array([])
    idx_m = np.array([])
    idx_o = np.arange(p)
    
    
    X_o = tf.gather(X, indices=idx_o, axis = 1)            
    
    mu_est_m = np.array([])
    eta_hat = _get_eta(X_o, beta,
                       idx_m, idx_o, 
                       mu_est_m)          
            
    #Calculates the gradient
    a_p = get_a_p(eta_hat, family)
    
    weights_1 = weights * (y - a_p)
    
    x = _update_x(X_o, idx_m, idx_o, 
                  mu_est_m,
                  weights = weights_1)        #p,N                        
    
    g = tf.math.reduce_sum(x, -1)                       #p,            
    
    #Calculates the Hessian
    a_pp = get_a_pp(eta_hat, family)
    weights_2 = weights * a_pp
    
    H_0 = _update_outer(X_o, idx_m, idx_o, 
                        mu_est_m,
                        weights = weights_2)
            
    H = -H_0
            
    if debug:
        print(idx_o)
        print(H.numpy()/N)
        print(g.numpy()/N)
#                import ipdb;ipdb.set_trace()
            
#            N_used += len(X_o)                    
    
#    direct = tf.linalg.inv(total_H) @ -tf.expand_dims(total_g,1)
#    direct = tf.squeeze(direct)
        
    try:
        H_inv = tf.linalg.inv(H)
    except:
        if bool_warn_inv:
#            import ipdb;ipdb.set_trace()
            eps = 1e-5
            print('Not invertible')
            H_inv = tf.linalg.inv(H+eps*tf.eye(p))
        else:
            raise ValueError
        
#    direct = tf.einsum('ij,j->i', tf.linalg.inv(total_H), -total_g)
    direct = tf.einsum('ij,j->i', H_inv, -g)
    
    return direct

def get_kl(X, beta_1, beta_2, family, 
           y = None):
        
    eta_1 = tf.einsum('ij,j->i', X, beta_1)
    eta_2 = tf.einsum('ij,j->i', X, beta_2)
    
    if family == 'lr':
        y_prob_1 = get_a_p(eta_1, family)
        y_prob_2 = get_a_p(eta_2, family)
        
        y_prob_1_in = tf.stack([1-y_prob_1,y_prob_1],1)
        y_prob_2_in = tf.stack([1-y_prob_2,y_prob_2],1)
        
        kl = tf.keras.losses.KLDivergence()(y_prob_1_in, y_prob_2_in)
    
    elif family == 'normal':
        var_1 = get_param(eta_1, y)
        var_2 = get_param(eta_2, y)
        
        kl_all = get_KL_uni(eta_1, var_1, 
                            eta_2, var_2)
        
        kl = tf.math.reduce_mean(kl_all)
    
    return kl

def get_param(eta, y):
    
    '''
    In:
        eta:    N,
        y:      N,
    '''
        
    err = y-eta
    var = tf.math.reduce_mean(err**2,0)
    
#    print(var/(np.std(y)**2))
    return var

def get_mice_beta(X, y, p_r_x, 
                  idx_adv,
                  family,
                  bool_o_mat,
                  bool_bias):
    
    '''
    In:
        X:              N,p     
        y:              N,
        
    '''
    N,p = X.shape
    
    if len(idx_adv) > 1:
        raise ValueError
        
    for idx_j in idx_adv:
        
        X_j = tf.gather(X, indices=idx_j, axis = 1)
        
        idx_rest = np.setdiff1d(np.arange(p), 
                                np.array([idx_j])).astype(np.int32)
        
        X_nj = tf.gather(X, indices=idx_rest, axis = 1)
        
        Z = tf.concat([X_nj,tf.expand_dims(y,1)],1)
        
        idx_adv_in = np.array([])
        
        beta_0 = tf.constant(tf.zeros(Z.shape[1]),
                             tf.float32) #/x_in.shape[0]
#        import ipdb;ipdb.set_trace()
        if np.array_equal(np.unique(X_j.numpy()),[0,1]):
            family_mice = 'lr'
        else:
            family_mice = 'normal'
            
        beta_j = get_irls_enum(Z, X_j, 
                              p_r_x, 
                              idx_adv_in,
                              family_mice,
                              bool_o_mat,
                              bool_bias,  
                              beta_0 = beta_0,
                              bool_while = True,
                              max_steps = 50,
                              type_impute = None,
                              bool_cca = True,
                              lmbda=1e-5,
                              bool_tf_inv = True
                              )
        if family_mice == 'normal':
            sigma_sq_j = get_dispersion(Z, X_j,
                                        beta_j,
                                        p_r_x, 
                                        idx_adv_in,
                                        bool_o_mat,
                                        family_mice,
                                        bool_bias,
                                        type_impute = None,
                                        kwargs_imp = {},
                                        bool_cca = True)
        elif family_mice == 'lr':
            sigma_sq_j = 1
            
        print('mice sigma_sq',sigma_sq_j)
            
    return beta_j, sigma_sq_j

def get_impute_mice(X, y,
                    idx_m,
                    beta_mice,
                    k,
                    sigma_sq_mice,
                    seed_mice):
    
    N,p = X.shape
    
    if len(idx_m) > 1:
        raise ValueError
                    
    if np.array_equal(np.unique(X.numpy()[:,idx_m[0]]),[0,1]):
        family_mice = 'lr'
    else:
        family_mice = 'normal'
        
    idx_rest = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
    
    X_nj = tf.gather(X, indices=idx_rest, axis = 1)
    
    Z = tf.concat([X_nj,tf.expand_dims(y,1)],1)
    eta = tf.einsum('ij,j->i', Z, beta_mice)
    
    X_j = tf.squeeze(tf.gather(X, indices=idx_m, axis = 1)) #N,
#    X_j = X.numpy()[:,idx_m[0]]
    
    if family_mice == 'normal':
#        err = np.mean((X_hat_m.numpy()[:,0]-X_j)**2)
        X_hat_m = tf.expand_dims(eta, 1)
        if 0:   

            tf.random.set_seed(seed_mice)
            X_hat_m = X_hat_m + \
                        tf.random.normal(X_hat_m.shape)*tf.math.sqrt(sigma_sq_mice)
            print('adding noise','seed',seed_mice)
    
    #            X_hat_m = tf.expand_dims(eta+tf.random.normal(eta.shape), 1)
#        print('nmse:', np.round(err.numpy()/np.std(X_j.numpy())**2,3))
        prob = None
                 
        
    elif family_mice == 'lr':
        
        prob_0 = get_a_p(eta, family_mice)
        auc = roc_auc_score(X_j.numpy(), prob_0.numpy())
        print('auc: %.3f'% auc)
        
        if k == 0:
            X_hat_m = tf.zeros(eta.shape[0],tf.float32)
            prob = 1-prob_0
        elif k == 1:
            X_hat_m = tf.ones(eta.shape[0],tf.float32)
            prob = prob_0
        
        X_hat_m = tf.expand_dims(X_hat_m, 1)
    
    return X_hat_m, prob
                
def get_impute(X, y,
               idx_o, idx_m,
               type_impute,
               idx_sub_m,
               mu_est_0 = None,
               beta_mice = None,
               sigma_sq_mice = None,
               k = -1,
               seed_mice = None):
    
    N,p = X.shape
    
    X_o = tf.gather(X, indices=idx_o, axis = 1)                                

    if len(idx_m) > 0:
        
        if type_impute == 'mean':
            mu_est_m = tf.gather(mu_est_0, idx_sub_m)
            X_hat_m = tf.repeat(tf.expand_dims(mu_est_m,0),
                               [X_o.shape[0]], 0) # N,|m|
            prob = None
            
        elif type_impute == 'mice':
            X_hat_m, prob = get_impute_mice(X, y,
                                            idx_m,
                                            beta_mice,
                                            k,
                                            sigma_sq_mice=sigma_sq_mice,
                                            seed_mice = seed_mice)
                
        
        X_hat_unsort = tf.concat([X_o, X_hat_m],1)
        idx = np.argsort(np.hstack([idx_o,idx_m]))
        X_hat = tf.gather(X_hat_unsort, idx,axis=1)
    else:
        X_hat = X_o
        prob = None
    
#    if not ((type_impute == 'mice') and (family_mice == 'lr')):
        
    return X_hat, prob

#def _get_eta_mice
def get_impute_reusable(X, y,
                        p_r_x,
                        idx_adv,
                        type_impute,
                        bool_o_mat,
                        bool_bias,
                        family,
                        verbose = 0):
    
    if type_impute == 'mean':
        X_mask = tf.gather(X, indices=idx_adv, axis = 1)
        
        mu_est_0 = helper_impute_tf.get_mu(X_mask, p_r_x, 
                                           bool_o_mat)
        kwargs_imp = dict(mu_est_0=mu_est_0)
        
        if verbose: print('mean_est',np.round(mu_est_0.numpy(),2))
        
    elif type_impute == 'mice':
        beta_mice, sigma_sq_mice = get_mice_beta(X, y, p_r_x, 
                                                 idx_adv, family,
                                                 bool_o_mat, bool_bias)
#        import ipdb;ipdb.set_trace()            
        print('beta_mice',np.round(beta_mice,1))
        kwargs_imp = dict(beta_mice=beta_mice,
                          sigma_sq_mice=sigma_sq_mice,
                          seed_mice=42)
            
    return kwargs_imp

def sub_enum(X, y,
              beta,
              p_r_x, 
              idx_adv,
              bool_o_mat,
              family,
              bool_bias,
              type_impute,
              debug = 0, 
              bool_full = True,
              verbose = 0,
              bool_warn_inv = False,
#                  bool_warn_inv = 1,
              bool_cca = False,
              lmbda = 0,
              return_grad = False,
              bool_solve_np = False,
              type_scaler = None):
    
    '''
    Modifies Algorithm 8.1 in \cite{Murphy}
    
    +
    In:
        X:          N,p
        idx_adv:    p_sub,        #masked variables
    Inter:
        total_inr:  #
        total_x:    #
    Out:
        direct:     #direction used in the irls
            
    '''    
    
#    import ipdb;ipdb.set_trace()
    N,p = X.shape
    
    if len(idx_adv) != 0:
        kwargs_imp = get_impute_reusable(X, y,
                                         p_r_x,
                                         idx_adv,
                                         type_impute,
                                         bool_o_mat,
                                         bool_bias,
                                         family,
                                         verbose=verbose
                                         )
    else:
        kwargs_imp = {}
        
        
    total_loss = tf.constant(0,tf.float32)
    
    if debug:
        p_r_x = tf.Variable(initial_value= tf.ones((N,p))/2, 
                          dtype='float32')    
    
#    idx_n_adv = np.setdiff1d(idx_adv, idx_m)
#            
#    N_used = 0
    p_sub = len(idx_adv)
    if not bool_full:            
        pass
        
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        if not bool_full:
            pass
            
        else:
            num = tf.gather(p_r_x, indices=np.array([i]), axis = 1)
            
            if ((p_sub == p) and not bool_bias) or \
               ((p_sub == p-1) and bool_bias):
#                if bool_bias: print('handling the bias term')
#                print('Possible error handling the bias term.')
                p_miss = tf.gather(p_r_x, indices=np.array([p_r_x.shape[1]-1]), 
                                   axis = 1)                
            else:
                p_miss = 0
            
            weights = tf.squeeze(num/(1-p_miss))                
        
        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
                
        if ((len(idx_o) != 0) and not bool_bias) or \
           ((len(idx_o) != 1) and bool_bias):
               
           #Checks at least one original feature is observed
               
#            if bool_bias: print(len(idx_o),'handling the bias term')

#            import ipdb;ipdb.set_trace()
            if debug_imp:
                X_o = tf.gather(X, indices=idx_o, axis = 1)
                mu_est_m = tf.gather(kwargs_imp['mu_est_0'], idx_sub_m)
            
            K = 2
            for k in range(K):
                X_hat_0, prob = get_impute(X, y, idx_o, idx_m,
                                         idx_sub_m = idx_sub_m,
                                         type_impute = type_impute,
                                         k = k,
                                         **kwargs_imp)
                
                if type_scaler is not None:
                    X_hat = get_scaled(X_hat_0, type_scaler)
                else:
                    X_hat = X_hat_0
                
                if prob is not None:
                    weights_1 = weights * prob
                else:
                    weights_1 = weights
                    
                X_j = tf.gather(X_hat, indices=idx_adv[0], axis = 1)
                eq_1 = tf.cast(X_j==1,tf.float32)
                
                if bool_cca:
                    A = p_r_x @ bool_o_mat        #N,p_sub        P_R_j|X_j(1;)
                    B = tf.math.reduce_mean(A,0)    #p_sub          P_R_j(1)
                
                eps = 1e-6
                if verbose: print('prior',B.numpy())
                loss = tf.math.reduce_mean(weights_1*eq_1)
                
                if bool_cca:
                    loss = loss/(B+eps)
                
                if k != 1:
                    print(idx_o,k,loss.numpy())  
                else:
                    print(idx_o,k,loss.numpy(),'and prob',prob.numpy().mean()) 
                
                total_loss = total_loss + loss
                
#                print('total_i',total_loss.numpy())
                                
                
                if prob is None:
                    break
            
        if bool_cca:
            break
        
    print('total',total_loss.numpy())
                
    return total_loss

def get_dispersion(X, y,
                  beta,
                  p_r_x, 
                  idx_adv,
                  bool_o_mat,
                  family,
                  bool_bias,
                  type_impute,
                  debug = 0, 
                  bool_full = True,
                  verbose = 0,
                  bool_cca = False,
                  kwargs_imp = None,
                  eps = 1e-6):
    
    if not bool_cca:
        raise ValueError

    N,p = X.shape
    
    if kwargs_imp is None:
        if len(idx_adv) != 0:
            kwargs_imp = get_impute_reusable(X, y,
                                             p_r_x,
                                             idx_adv,
                                             type_impute,
                                             bool_o_mat,
                                             bool_bias,
                                             family,
                                             verbose=verbose
                                             )
        else:
            kwargs_imp = {}            
    
    if debug:
        p_r_x = tf.Variable(initial_value= tf.ones((N,p))/2, 
                          dtype='float32')    
    

    p_sub = len(idx_adv)
    if not bool_full:            
        pass
        
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        if not bool_full:
            pass
            
        else:
            num = tf.gather(p_r_x, indices=np.array([i]), axis = 1)
            
            if ((p_sub == p) and not bool_bias) or \
               ((p_sub == p-1) and bool_bias):
#                if bool_bias: print('handling the bias term')
#                print('Possible error handling the bias term.')
                p_miss = tf.gather(p_r_x, indices=np.array([p_r_x.shape[1]-1]), 
                                   axis = 1)                
            else:
                p_miss = 0
            
            weights = tf.squeeze(num/(1-p_miss))                
        
        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
                
        if ((len(idx_o) != 0) and not bool_bias) or \
           ((len(idx_o) != 1) and bool_bias):
               
           #Checks at least one original feature is observed
               
#            if bool_bias: print(len(idx_o),'handling the bias term')

#            import ipdb;ipdb.set_trace()
            if debug_imp:
                X_o = tf.gather(X, indices=idx_o, axis = 1)
                mu_est_m = tf.gather(kwargs_imp['mu_est_0'], idx_sub_m)
            
            K = 2
            for k in range(K):
                X_hat, prob = get_impute(X, y, idx_o, idx_m,
                                   idx_sub_m = idx_sub_m,
                                   type_impute = type_impute,
                                   k=k,
                                   **kwargs_imp)
                     
                if prob is not None:
                    weights_0 = weights * prob
                else:
                    weights_0 = weights
                                        
                eta_hat = tf.einsum('ij,j->i', X_hat, beta) 
                if debug_imp:
                    eta_hat2 = _get_eta(X_o, beta,
                                       idx_m, idx_o, 
                                       mu_est_m)
                    if not np.allclose(eta_hat.numpy(),eta_hat2.numpy()): raise ValueError
                
#                import ipdb;ipdb.set_trace()
                #Calculates the gradient
                if family == 'normal':
                    err_0 = (y-eta_hat)**2
                    err_1 = tf.math.reduce_sum(err_0*weights_0)
                    
                    total_weight = tf.math.reduce_sum(weights_0)
                    
                    sigma_sq = err_1/(total_weight+eps)                
                
                if prob is None:
                    break
        
        if bool_cca:
            break    
        
    return sigma_sq