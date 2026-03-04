import numpy as np
import tensorflow as tf
from scipy.special import expit
#from helper_scale import get_scaled_sklearn
import helper_impute_tf
from External.MissingDataOT.utils_np import _MNAR_self_mask_logistic, _MNAR_mask_logistic

class SelfMasking():
    
    def __init__(self,
                 idx_mask,
                 coeffs,
                 intercepts,
                 self_mask):
        
        self.idx_mask = idx_mask
        self.coeffs = coeffs
        self.intercepts = intercepts
        self.self_mask = self_mask
    
    def __call__(self, X, training = False):
                
        return get_prob(X, self.idx_mask, 
                        self.coeffs, self.intercepts,
                        self.self_mask)

def get_prob(Z_0,
             idx_mask,
             coeffs,
             intercepts,
             self_mask,
             seed = 42):
    
    '''
    Inter:
        mar_prob:      N,p_sub     #probability each mask is observed   
    '''
    p = Z_0.shape[1]
    p_sub = len(idx_mask)
    
    if p_sub != p:
        n_out = int(2**p_sub)
    else:#All missing not feasible
        n_out = int(2**p_sub)-1
        import ipdb;ipdb.set_trace()
        
    obs_mat = helper_impute_tf.get_obs_mat_wrap(p_sub, n_out)
    
#    import ipdb;ipdb.set_trace()
    
    if self_mask == 1:
        idx_input = idx_mask
        Z = Z_0[:,idx_mask]
        p_r_0 = expit(Z * coeffs + intercepts)
    else:
        if self_mask == 0:
            idx_input = np.append(idx_mask, p-1)        
        if self_mask == -1:
            idx_input = np.append(idx_mask, [p-2,p-1])
        Z = Z_0[:,idx_input]
        p_r_0 = expit(Z@coeffs + intercepts)
        
    p_r_1 = 1- p_r_0        
    
    log_prob = tf.math.log(p_r_1)
    log_1_prob = tf.math.log(1-p_r_1)
    
    temp = tf.einsum('ij,kj->ik', log_prob, obs_mat)+\
            tf.einsum('ij,kj->ik', log_1_prob, 1-obs_mat)
            
    prob = tf.math.exp(temp)
            
    return prob

def get_baseline_MNAR(X_train_all,
                      y_train_all,
                      idx_mask,
                      bool_scale_all,
                      seed_model,
                      missing_rate,
                      self_mask):
    
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
    
    if self_mask == -1:
        scale_mu[-1] = 0
        scale_std[-1] = 1
        
    scale_mu = np.append(scale_mu,[mu_y])
    scale_std = np.append(scale_std,[std_y])

    Z = np.concatenate([X_train_all,y_train_all[:,np.newaxis]],1)
    
    Z_scaled = (Z - scale_mu)/scale_std
    
#    import ipdb;ipdb.set_trace()
    
    if self_mask == 1:
        coeffs, intercepts = _MNAR_self_mask_logistic(Z_scaled[:,idx_mask], 
                                                      missing_rate, 
                                                      seed_model)
    else:
        if self_mask == 0:
            idx_input = np.append(idx_mask, Z.shape[1]-1)
            
        elif self_mask == -1:
            idx_input = np.append(idx_mask, [Z.shape[1]-2,Z.shape[1]-1])
            
        coeffs, intercepts = _MNAR_mask_logistic(Z_scaled[:,idx_input], 
                                                 missing_rate, 
                                                 seed_model,
                                                 np.arange(len(idx_mask)))
#        import ipdb;ipdb.set_trace()
    
    model = SelfMasking(idx_mask, coeffs, intercepts, self_mask)
    
    return model, scale_mu, scale_std

def get_mnar_params(omit_prop):
    
    rate_dict = {1.:0.072,
                .75:0.148,
                .50:0.485,
                .25:0.501
                }

    
    temp_dict = dict(
                seed_model=0,
#                seed_model=2,
#                self_mask=False,
                self_mask=-1,
                type_scaler='minmax',
                bool_scale_all=False,
#                missing_rate = 0.072
#                missing_rate = 0.25,
                missing_rate = rate_dict[omit_prop],
                )
    
#    type_mask = 'outcome'
    
#    if type_mask != 'outcome'
#    temp_dict['type_mask'] = type_mask
    
    return temp_dict