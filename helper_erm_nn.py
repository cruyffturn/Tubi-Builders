# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import numpy as np

from helper_em_tf import powerset
from helper_tf_glm import get_impute, get_impute_reusable
#from helper_impute_tf import get_obs_mat_wrap
import helper_tf_glm
#import helper_tf_model_irls
#def get_loss_enum(p_r_x, loss):
debug_solver = 0

from helper_scale import get_scaled

#family = 'reg'_set_param
def get_flat(params): 
    temp = [tf.reshape(param, [-1]) for param in params]
    params = tf.concat(temp, axis=0) #gpt    
    return params

def get_flat_2d(params): 
    temp = [tf.reshape(param, [param.shape[0],-1]) for param in params]
    params = tf.concat(temp, axis=1) #gpt    
    return params
        
def get_f_tilde(X, y,
            inner_model, 
            p_r_x,
            idx_adv,
#            bool_o_mat,
            type_impute,
            kwargs_imp,
            family,
            debug = 0, 
            bool_full = True,
            verbose = 0,
            bool_cca = False,
            debug_imp = 0,
            bool_mean = True):
    
    '''
    Computes f_tilde
    
    In:
        X:          N,p
        idx_adv:    p_sub,        #masked variables
    '''    
    
    bool_bias = False
#    import ipdb;ipdb.set_trace()
    N,p = X.shape
    
    if kwargs_imp is None:
#        if len(idx_adv) != 0:
#            kwargs_imp = get_impute_reusable(X, y,
#                                             p_r_x,
#                                             idx_adv,
#                                             type_impute,
#                                             bool_o_mat,
#                                             bool_bias,
#                                             family,
#                                             verbose=verbose
#                                             )
#        else:
#            kwargs_imp = {}
        raise ValueError
        
    if debug:
        p_r_x = tf.Variable(initial_value= tf.ones((N,p))/2, 
                          dtype='float32')    
    
    p_sub = len(idx_adv)
    if not bool_full:            
        pass
        
    total_loss = tf.constant(tf.zeros(1),tf.float32)
    
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        if i == tf.shape(p_r_x)[1]:
            if p_r_x.numpy().shape[1] != int(2**p_sub)-1:
                import ipdb;ipdb.set_trace()
            else:
                break
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        if not bool_full:
            pass
            
        else:
            num = tf.gather(p_r_x, indices=np.array([i]), axis = 1)
            
            if ((p_sub == p) and not bool_bias) or \
               ((p_sub == p-1) and bool_bias):
                  
                if p_r_x.numpy().shape[1] == int(2**p_sub):
                    import ipdb;ipdb.set_trace()
                    p_miss = tf.gather(p_r_x, indices=np.array([p_r_x.shape[1]-1]), 
                                       axis = 1)
                else:
#                    print('all missing is infeasible')
                    p_miss = 0
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
                X_hat, prob = get_impute(X, y, idx_o, idx_m,
                                   idx_sub_m = idx_sub_m,
                                   type_impute = type_impute,
                                   k = k,
                                   **kwargs_imp)
                                
                y_pred = inner_model(X_hat, training=True)  # Forward pass
                y_in = tf.expand_dims(y,1)
                
                if family == 'clsf':
                    loss_0 = tf.keras.losses.binary_crossentropy(y_in, y_pred,
                                                                 from_logits=True)
                else:
                    loss_0 = tf.keras.losses.mse(y_in, y_pred)
                    
                if prob is not None:
                    weights_0 = weights * prob
                else:
                    weights_0 = weights                
                
                loss_1 = loss_0 * weights_0
                
#                import ipdb;ipdb.set_trace()
                if bool_cca:
                    if not bool_mean:
                        loss = -tf.math.reduce_sum(loss_1)/tf.math.reduce_mean(weights_0)
                    else:
#                        print('using avg. loss')
                        loss = -tf.math.reduce_mean(loss_1)/tf.math.reduce_mean(weights_0)
                else:
                    if not bool_mean:
                        loss = -tf.math.reduce_sum(loss_1)
                    else:
                        loss = -tf.math.reduce_mean(loss_1)
                        
                total_loss = total_loss + loss
                
                if prob is None:
                    break
        
        if bool_cca:
#            print('ll cca')
            break
    
#    import ipdb;ipdb.set_trace()
    # Get regularization losses (if any)
    if len(inner_model.losses) != 0:
        
        reg_loss = tf.add_n(inner_model.losses)            
        total_loss = total_loss + reg_loss
        
    return tf.squeeze(total_loss) 

#@tf.function
def _get_f_y_fyy(X, y,
                 inner_model,
                 p_r_x, 
                 idx_adv,
                 type_impute,
                 family,
                 debug = 0, 
                 bool_full = True,
                 verbose = 0,
                 bool_cca = False,
                 kwargs_imp = None):
    
    '''
    Computes the gradient and hessian of the f_tilde w.r.t. \theta
    '''
    params = inner_model.trainable_variables
    
    with tf.GradientTape() as tape_i:
        tape_i.watch(params)
        with tf.GradientTape() as tape_ii:
            tape_ii.watch(params)
                                         
            f_tilde = get_f_tilde(X, y, inner_model, 
                                  p_r_x,
                                  idx_adv,
                                  type_impute,
                                  debug = debug, 
                                  bool_full = bool_full,
                                  verbose = verbose,
                                  bool_cca = bool_cca,
                                  kwargs_imp = kwargs_imp,
                                  family = family)
            
            g_l = tape_i.gradient(f_tilde, params)
            g = get_flat(g_l)
    
        H_l = tape_ii.jacobian(g, params)
        H = get_flat_2d(H_l)
    
    return g, H

#%%
def _get_solver(p, solver_layers, 
               solver_arch,
               seed_solver = 42,
               solver_lambda = 0,
               bool_tilde = True):
    
    print('using solver layers', solver_layers)
    
    if seed_solver is not None:
        tf.random.set_seed(seed_solver)
    
    if solver_lambda == 0:
        kernel_regularizer = None
        out_regularizer = None
        recurrent_regularizer = None
    else:
        kernel_regularizer = regularizers.L2(solver_lambda)
        out_regularizer = regularizers.L2(solver_lambda)
        recurrent_regularizer = regularizers.L2(solver_lambda)
    
    if solver_arch == 'mlp':
        if not debug_solver:
            
            if bool_tilde:
                model = InnerSolver()
            else:
                model = keras.Sequential()
                
            model.add(keras.Input(shape=(p,)))
            
            for unit in solver_layers:
                
                model.add(layers.Dense(unit, activation='sigmoid',
                                       kernel_regularizer=kernel_regularizer))
            
            model.add(layers.Dense(1,activation=None,
                                   kernel_regularizer=out_regularizer))
            
        else:
            if bool_tilde:
                
                model = InnerSolver([keras.Input(shape=(p,)),
                                  layers.Dense(1,activation=None,use_bias=False)
                              ])
    elif solver_arch == 'rnn':
        print('using solver_arch', solver_arch)
        if bool_tilde:        
            model = InnerSolver()
        else:
            model = keras.Sequential()
    
        
        model.add(keras.Input(shape=(p,)))
        
#        if solver_inshape is None:
        def expand_dims_layer(x):
            return tf.expand_dims(x, axis=2)
        model.add(layers.Lambda(expand_dims_layer))
#        else:
#            model.add(layers.Reshape(solver_inshape))
            
        model.add(layers.SimpleRNN(solver_layers[0], 
#                                   activation='sigmoid',
                                   activation='tanh', 
                                   kernel_regularizer=kernel_regularizer,
                                   recurrent_regularizer=recurrent_regularizer,
                                   ))
        
#        model.add(layers.LSTM(solver_layers[0], 
##                                   activation='sigmoid',
#                                   activation='tanh', 
#                                   kernel_regularizer=kernel_regularizer,
#                                   recurrent_regularizer=recurrent_regularizer,
#                                   stateful=False
##                                   input_shape=(1, p),
##                                   unroll = True
#                                   ))
                          
#        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1,activation=None,
                               kernel_regularizer=out_regularizer))
    elif solver_arch == 'rnn_p2012':
        print('using solver_arch', solver_arch)
        if bool_tilde:        
            model = InnerSolver()
        else:
            model = keras.Sequential()
    
        model.add(keras.Input(shape=(p,)))        
        model.add(layers.Reshape((48,10)))            
        model.add(layers.SimpleRNN(solver_layers[0],
                                   kernel_regularizer=kernel_regularizer,
                                   recurrent_regularizer=recurrent_regularizer,
                                   ))
        model.add(layers.Dropout(0.5))    
        model.add(layers.Dense(1,activation=None,
                               kernel_regularizer=out_regularizer))
    elif solver_arch == 'rnn_p2012_all':
        print('using solver_arch', solver_arch)
        if bool_tilde:        
            model = InnerSolver()
        else:
            model = keras.Sequential()
    
        model.add(keras.Input(shape=(p,)))
        model.add(layers.Reshape((48,37)))
        model.add(layers.SimpleRNN(solver_layers[0],
                                   kernel_regularizer=kernel_regularizer,
                                   recurrent_regularizer=recurrent_regularizer,
                                   ))
#        model.add(layers.Dropout(0.5))    
        model.add(layers.Dense(1,activation=None,
                               kernel_regularizer=out_regularizer))    
        
    return model

def get_solver(p, solver_layers, 
               solver_arch,
               seed_solver = 42,
               solver_lambda = 0,
#               solver_inshape = None,
               prev_solution = None):
    
    '''
    Initializes the inner solver
    '''

    model = _get_solver(p, solver_layers, 
                        solver_arch,
                        seed_solver = seed_solver,
                        solver_lambda = solver_lambda,
#                        solver_inshape = solver_inshape
                        )
#    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    #Compile might be unnesseary
#    print('maybe I need to adam')
    model.compile(
#                    loss=loss_fn,
#                  optimizer='adam',
                  run_eagerly = 1,
                  )
    
    if prev_solution is not None:
#        print('warm_starting setting weights')
        model.set_weights(prev_solution.get_weights())

    return model
            
def solve_y(outer_model, z_sub, 
            x_in, y,
            idx_mask_in,
            n_return_states = 0):
    
    '''
    Solves the inner problem
    '''
    
    p_r_x = outer_model(z_sub, training=True)  # Forward pass                                    
    #get p_r_x
    
    if outer_model.type_modeler != 'cca':            
        kwargs_imp = get_impute_reusable(x_in, y,
                                         p_r_x,
                                         idx_adv = idx_mask_in,
                                         type_impute = outer_model.type_impute,
                                         bool_o_mat = outer_model.bool_o_mat,
                                         bool_bias = True,
                                         family = None)
    else:
        kwargs_imp = {}
    
    if (len(outer_model.kwargs_solver['solver_layers']) != 0):
        solver, metrics_inner = _solve_y(x_in, y, p_r_x,
                          idx_mask_in,
                          outer_model.type_impute, 
                          outer_model.type_modeler,
                          kwargs_imp,
                          prev_solution = outer_model.prev_solution,
                          n_return_states = n_return_states,
                          family = outer_model.family,
                          type_scaler = outer_model.type_scaler,
                          **outer_model.kwargs_solver)
    else:
#         or \
#        (outer_model.type_modeler != 'cca'):
        if 0:
            solver, metrics_inner = _solve_y_linear(x_in, y, p_r_x,
                                                idx_mask_in,
                                                outer_model.type_impute, 
                                                outer_model.type_modeler,
                                                kwargs_imp,
                                                prev_solution = outer_model.prev_solution,
                                                n_return_states = n_return_states,
                                                family = outer_model.family,
                                                **outer_model.kwargs_solver)
        else:
            solver, metrics_inner = _solve_y_linear_irls(outer_model, x_in, y, 
                                                         p_r_x)
            
            if outer_model.bool_resolve:
                bias = np.log10(np.abs(solver.get_weights()[-1][0]))
    #            import ipdb;ipdb.set_trace()
                if bias > 10:
                    print('resolving')
                    solver, metrics_inner = _solve_y_linear_irls(outer_model, x_in, y, 
                                                                 p_r_x,
                                                                 lmbda = 1e-3)
                
        
    return solver, metrics_inner

def _solve_y(x_in, y, p_r_x,
             idx_mask_in,
             type_impute, type_modeler,
             kwargs_imp,
             family,
             type_scaler,
             solver_layers, 
             solver_epochs,
             solver_lr,
             solver_seed = 42,
             solver_arch = 'mlp',
             solver_early = False,
             solver_warm_start = False,
             solver_lambda = 0,
             solver_optimizer = 'adam',
             solver_optimizer_kwargs = {},
             n_return_states = 0,
#             solver_inshape = None,
             prev_solution = None):
    
        
    solver = get_solver(x_in.shape[1],
                        solver_layers, 
                        solver_arch,
                        solver_seed,
                        solver_lambda,
#                        solver_inshape,
                        prev_solution
                        )
    
    solver._set_param(
                   p_r_x,
                   idx_adv = idx_mask_in,
                   type_impute = type_impute,
                   kwargs_imp = kwargs_imp,
                   type_modeler = type_modeler,
                   family = family,
                   )
    
    
    if not solver_early:
        if solver_arch != 'p2012':

            if n_return_states == 0:
                
                inner_loss = custom_fit(solver, x_in, y, 
                                        solver_epochs,
                                        solver_lr,
                                        solver_optimizer,
                                        prev_solution,
                                        **solver_optimizer_kwargs)
#                pass
            
            else:
#                solver = [solver]*n_return_states
                inner_loss, solver_l = custom_fit_with_return(solver, x_in, y, 
                                                             solver_epochs,
                                                             solver_lr,
                                                             solver_optimizer,
                                                             prev_solution,
                                                             n_return_states,)
#                import ipdb;ipdb.set_trace()
#                solver = [solver]*n_return_states
#                solver = [solver_l[-1]]*n_return_states
                solver = solver_l
                
        else:
            inner_loss = custom_fit_batch(solver, x_in, y, 
                             solver_epochs, solver_lr,
                             prev_solution)
    else:
        inner_loss = custom_fit_early(solver, x_in, y, 
                         solver_epochs,
                         solver_lr,
                         prev_solution=prev_solution
#                         min_delta = 1e-4,
#                         patience = 1
                         )
    
    metrics_inner = dict(inner_loss=inner_loss)
                         
    return solver, metrics_inner
    
def custom_fit(model, x, y, 
               epochs, lr,
               name_optimizer,
               prev_solution,
               **optimizer_kwargs):
    
    optimizer = get_optimizer(lr, name_optimizer, 
                              prev_solution,
                              **optimizer_kwargs)

    if epochs == 0:
        loss_value = None
        
    for epoch in range(epochs):
#        print("\nStart of epoch %d" % (epoch,))
    
        with tf.GradientTape() as tape:
            loss_value = get_f_tilde(x, y,
                            model, 
                            model._p_r_x,
                            model._idx_adv,
                            model._type_impute,
                            model._kwargs_imp,
                            bool_cca = model._type_modeler == 'cca',
                            family = model._family
                            )
            loss_value=-loss_value
    
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))            
        #Log every 200 batches.
        
        if (epoch % 100 == 0) and 0:
            print('inner_grad',np.linalg.norm(tf.squeeze(grads[0]),1))
#                print(
#                    "Training loss (for one batch) at step %d: %.4f"
#                    % (step, float(loss_value))
#                )
#                print("Seen so far: %s samples" % ((step + 1) * batch_size))   
            
#    print('inner_grad',np.linalg.norm(tf.squeeze(grads[0]),1))
#    import ipdb;ipdb.set_trace()
            
    return loss_value

def custom_fit_early(model, x, y, 
                     epochs, lr,
                     prev_solution,
                     min_delta = 1e-4,
                     patience = 10):
    
    '''
    @chatgpt
    Early stopping
    '''
    
#    epochs = 10
#    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    #    optimizer = keras.optimizers.Adam(learning_rate=lr)
    if prev_solution is not None:
#        optimizer.set_weights(prev_solution.optimizer.get_weights())
        optimizer = prev_solution.optimizer
        print('using prev optimizer')
#        optimizer = prev 
#        import ipdb;ipdb.set_trace()
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        
    best_metric = float("inf")
    wait = 0  # Counter for early stopping  
    
    for epoch in range(epochs):
        
    
        with tf.GradientTape() as tape:
            loss_value = get_f_tilde(x, y,
                            model, 
                            model._p_r_x,
                            model._idx_adv,
                            model._type_impute,
                            model._kwargs_imp,
                            bool_cca = model._type_modeler == 'cca',
                            family = model._family
                            )
            loss_value=-loss_value
    
#        print("\nStart of epoch %d" % (epoch,),'%.2f'%loss_value)
        
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))            
        #Log every 200 batches.
        
#        import ipdb;ipdb.set_trace()
#        np.max(grads)
#        metric = np.max([np.max(np.abs(grad)) for grad in grads])
#        metric = np.sum([np.sum(np.abs(grad)) for grad in grads])
        metric = loss_value
        # Check if loss has improved enough
        if 1:
            if best_metric - metric > min_delta:
    #            best_loss = loss_value
                best_metric = metric
                wait = 0  # Reset wait period if there is an improvement
            else:
                wait += 1
    #            print(f"No significant improvement for {wait} epoch(s).")
            
            if wait >= patience:
    #            print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
    
    if wait >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}.")
    else:
        print("Stop maximum epoch")
        
#    print('inner_grad',np.linalg.norm(tf.squeeze(grads[0]),1))
#    import ipdb;ipdb.set_trace()
            
    return loss_value

def custom_fit_batch(model, x, y, 
               epochs, lr,
               prev_solution):
    
    '''
    @gemini
    '''
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    BATCH_SIZE = 128
#    EPOCHS = 5
    num_samples = x.shape[0]
    num_batches = num_samples // BATCH_SIZE

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for batch_index in range(num_batches):
            start_index = batch_index * BATCH_SIZE
            end_index = (batch_index + 1) * BATCH_SIZE
            x_batch = x[start_index:end_index]
            y_batch = y[start_index:end_index]
            prob_batch = model._p_r_x[start_index:end_index]        

            with tf.GradientTape() as tape:
                loss_value = get_f_tilde(x_batch, y_batch,
                                model, 
                                prob_batch,
                                model._idx_adv,
                                model._type_impute,
                                model._kwargs_imp,
                                bool_cca = model._type_modeler == 'cca',
                                family = model._family
                                )
                loss_value=-loss_value
                
            if batch_index % 100 == 0:
                print(f"  Batch {batch_index}, Loss: {loss_value.numpy():.4f}")            
    
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))            
        
    return loss_value

class InnerSolver(keras.Sequential):
              
    '''
    Extends the keras.Sequential for implementing a custom loss function
    TBD maybe use a custom fit function
    '''
    
    def _set_param(self,
                   p_r_x,
                   idx_adv,
                   type_impute,
                   kwargs_imp,
                   type_modeler,
                   family,
                   ):    
            
        self._p_r_x = p_r_x
        self._idx_adv = idx_adv
        self._type_impute = type_impute
        self._kwargs_imp = kwargs_imp
        self._type_modeler = type_modeler        
        self._family = family


def _get_fyy_mvp(X, y,
                 inner_model,
                 p_r_x, 
                 idx_adv,
                 type_impute,
                 alpha,
                 family,
                 debug = 0, 
                 bool_full = True,
                 verbose = 0,
                 bool_cca = False,
                 kwargs_imp = None):
    
    '''
    Computes the matrix vector product of llp Hessian with vector alpha
    '''
    params = inner_model.trainable_variables
    
    with tf.GradientTape() as tape_i:
        tape_i.watch(params)
        
        with tf.GradientTape() as tape_ii:
            tape_ii.watch(params)
                                         
            f_tilde = get_f_tilde(X, y, inner_model, 
                                  p_r_x,
                                  idx_adv,
                                  type_impute,
                                  debug = debug, 
                                  bool_full = bool_full,
                                  verbose = verbose,
                                  bool_cca = bool_cca,
                                  kwargs_imp = kwargs_imp,
                                  family = family,
                                  type_scaler = type_scaler)
            
            f_tilde=-f_tilde
            
            g_l = tape_i.gradient(f_tilde, params)
            g = get_flat(g_l)
    
#            import ipdb;ipdb.set_trace()
            mvp_l = tape_ii.gradient(g, params, output_gradients=alpha)
            mvp = get_flat(mvp_l)
    
    return g, mvp

def custom_fit_with_return(model, x, y, 
                           epochs, lr,
                           name_optimizer,
                           prev_solution,
                           K):
    
#    model_list_callback = ModelListCallback(K)
    model_l = []
    
#    if prev_solution is not None:
#        print('using prev optimizer')
#    else:
    optimizer = get_optimizer(lr, name_optimizer, 
                              prev_solution)

    if epochs <= K:
        model_copy = tf.keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        model_l.append(model_copy)

    for epoch in range(epochs):
#        print("\nStart of epoch %d" % (epoch,))
    
        with tf.GradientTape() as tape:
            loss_value = get_f_tilde(x, y,
                            model, 
                            model._p_r_x,
                            model._idx_adv,
                            model._type_impute,
                            model._kwargs_imp,
                            bool_cca = model._type_modeler == 'cca',
                            family = model._family,
                            )
            loss_value=-loss_value
    
#        import ipdb;ipdb.set_trace()
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))            
        #Log every 200 batches.
        
        if epoch >= (epochs-K):
            model_copy = tf.keras.models.clone_model(model)
            model_copy.set_weights(model.get_weights())
            model_l.append(model_copy)

        if (epoch % 100 == 0) and 0:
            print('inner_grad',np.linalg.norm(tf.squeeze(grads[0]),1))
#                print(
#                    "Training loss (for one batch) at step %d: %.4f"
#                    % (step, float(loss_value))
#                )
#                print("Seen so far: %s samples" % ((step + 1) * batch_size))   
            
#    print('inner_grad',np.linalg.norm(tf.squeeze(grads[0]),1))
#    import ipdb;ipdb.set_trace()
            
    return loss_value, model_l

def get_optimizer(lr, name_optimizer, 
                  prev_solution,
                  **optimizer_kwargs):
    
    if name_optimizer == 'adam':
        if prev_solution is None:
            optimizer = keras.optimizers.Adam(learning_rate=lr,
                                              **optimizer_kwargs)
        else:
    #        optimizer = prev_solution.optimizer
            raise ValueError
            
    elif name_optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr,
                                         **optimizer_kwargs)
        print('using sgd')
    
    return optimizer

def _solve_y_linear(x_in, y, p_r_x,
             idx_mask_in,
             type_impute, type_modeler,
             kwargs_imp,
             solver_layers, 
             solver_epochs,
             solver_lr,
             solver_seed = 42,
             solver_arch = 'mlp',
             solver_early = False,
             solver_warm_start = False,
             solver_lambda = 0,
             solver_optimizer = 'adam',
             solver_optimizer_kwargs = {},
             n_return_states = 0,
#             solver_inshape = None,
             prev_solution = None):
    
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
#    use_bias = True
    if family == 'clsf':        
        model = LogisticRegression(solver='saga', 
                                   random_state=solver_seed, 
                                   max_iter=solver_epochs)
        if type_modeler == 'cca':
            model.fit(x_in, y, sample_weight = p_r_x.numpy()[:,0])
        beta = model.coef_[0]
        bias = model.intercept_
    else:
#        import ipdb;ipdb.set_trace()
        model = LinearRegression()
        if type_modeler == 'cca':
            model.fit(x_in, y, sample_weight = p_r_x.numpy()[:,0])
        else:
#            x_0 = x_in
#            x_1 = tf.stack([x_in,)
            model.fit(x, y, sample_weight = p_r_x.numpy()[:,0])
                    
            
        beta = model.coef_
        bias = model.intercept_

#        model_wrap = keras.Sequential(layers.Dense(units=1,use_bias=use_bias,
#                       kernel_initializer=keras.initializers.Constant(model.coef_[0]),
#                               ))
        
        model_wrap, metrics_inner = _get_linear_wrapped(beta, bias, x_in) 
    
    return model_wrap, metrics_inner

def _solve_y_linear_irls(outer_model, x_in, y, p_r_x,
                         lmbda = 0):
    
    n_steps = None
    max_steps = 20
#    max_steps = 50
    x_in_bias = tf.concat([tf.ones((x_in.shape[0],1)),x_in],
                          axis=1)
    bool_bias_inner = True
    idx_mask_in = outer_model.idx_mask+1    
    beta_0 = tf.constant(tf.zeros(x_in_bias.shape[1]),
                         tf.float32) #/x_in.shape[0]

#    import ipdb;ipdb.set_trace()
    family_in = {'reg':'normal',
                 'clsf':'lr'}[outer_model.family]
    beta_1 = helper_tf_glm.get_irls_enum(x_in_bias, y, 
                                       p_r_x,     
                                         idx_mask_in,
                                         family_in,
                                         outer_model.bool_o_mat,
                                         bool_bias_inner,
                                         beta_0 = beta_0,
                                         n_steps = n_steps,
    #                                     n_steps = 1,
    #                                     bool_while = False
                                         bool_while = True,
                                         max_steps = max_steps,
                                         bool_cca = outer_model.type_modeler == 'cca',
                                         bool_solve_np = True,
                                         type_impute = outer_model.type_impute,
                                         type_scaler = outer_model.type_scaler,
                                         lmbda = lmbda
                                         )
#    import ipdb;ipdb.set_trace()
    beta = tf.gather(beta_1,np.arange(1,x_in.shape[1]+1))
    bias = tf.gather(beta_1,np.array([0]))
    
    model_wrap, metrics_inner = _get_linear_wrapped(beta, bias, x_in,
                                                    outer_model.type_scaler)
    
    return model_wrap, metrics_inner

def _get_linear_wrapped(beta, bias, x_in,
                        type_scaler):
    
    dense_layer = layers.Dense(units=1,
                   kernel_initializer=keras.initializers.Constant(beta),
                   bias_initializer=keras.initializers.Constant(bias),
                                   )
    if type_scaler is None:
        model_wrap = keras.Sequential(dense_layer)
    else:
        scale_layer = keras.layers.Lambda(lambda x: get_scaled(x, type_scaler))
        model_wrap = keras.Sequential([scale_layer,dense_layer])
        
    metrics_inner = {}
    
#    import ipdb;ipdb.set_trace()

    model_wrap(x_in)

    return model_wrap, metrics_inner