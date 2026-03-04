'''
Configuration for the experiments
'''
import os
import numpy as np
import inspect
import pickle

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
sharedPath = os.path.join(os.environ['SHARED'],
                              'Code',
                              currPath.split(os.environ['CODE_PATH'])[1][1:])

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

#from helper_glm_main import get_data as _get_data
#import helper_save
import helper_ate
def get_data(name, return_bias = 0):
    
    test_size = .2
    
    if name == 'moons':
    
        #%%
        X_0, y = make_moons(noise=0.3, random_state=0)
        
        X = StandardScaler().fit_transform(X_0)
        feat_names = ['feature_1','feature_2']
        bool_bias = 1
    
        X_train_all, X_test, \
        y_train_all, y_test = train_test_split(X, y, test_size = test_size, 
                                           random_state = 42)
    elif name == 'p2012':
        savePath = os.path.join(os.environ['SHARED'],
                                  'Code','ts','data','set-a.p')
        
        with open(savePath, "rb") as f:
            data = pickle.load(f)
            
        # prepare the dataset
#        data = preprocess_physionet2012(subset='set-a')
        X_train_0, X_val_0 = data["train_X"], data["val_X"]
        y_train_all, y_val = data["train_y"], data["val_y"]
        
        X_train_0[np.isnan(X_train_0)] = 0
        X_val_0[np.isnan(X_val_0)] = 0
        
        idx_sub = np.array([10, 33, 30, 29, 20, 8, 4, 14, 17, 9])
        X_train_1 = X_train_0[:,:,idx_sub]
        X_val_1 = X_val_0[:,:,idx_sub]
        
        X_train_all = X_train_1.reshape(X_train_0.shape[0],-1)
        X_val = X_val_1.reshape(X_val_0.shape[0],-1)
        
        X_test = X_val
        y_test = y_val
        
        feat_names = np.array(data['feature_names'])[idx_sub].tolist()
        bool_bias = 1
#        X_test_0[np.isnan(X_test_0)] = 0

    elif name == 'p2012_all':
        savePath = os.path.join(os.environ['SHARED'],
                                'Code','ts','data')        
        
        for str_ in ['a','b','c']:
                
            with open(os.path.join(savePath,'physio2012_set-%s.p'%str_), "rb") as f:
                data = pickle.load(f)
            
            X_train_0 = np.concatenate([data["train_X"], data["val_X"]])
            y_train_all = np.concatenate([data["train_y"], data["val_y"]])
                        
            X_test_0 = data["test_X"]
            y_test = data["test_y"]
            
        #    y_train_all, y_val = data["train_y"], data["val_y"]
        N, T, p = X_train_0.shape
        
        #%%
        X_train_0[np.isnan(X_train_0)] = 0
#        import ipdb;ipdb.set_trace()
        #X_val_0[np.isnan(X_val_0)] = 0
        X_test_0[np.isnan(X_test_0)] = 0
        
#        idx_sub = np.array([10, 33, 30, 29, 20, 8, 4, 14, 17, 9])
#        X_train_0 = X_train_0[:,:,idx_sub]
#        X_val_0 = X_val_0[:,:,idx_sub]
        
#        p = idx_sub


        X_train_all = X_train_0.reshape(-1,int(p*T))
        X_test = X_test_0.reshape(-1,int(p*T))
        
        feat_names = np.array(data['feature_names']).tolist()
        
    elif name == 'cali':
        X_train_all, X_test, \
        y_train_all_0, y_test_0,\
        feat_names = _get_data(name)
        
        y_train_all = y_train_all_0>np.median(y_train_all_0, 0)
        y_test = y_test_0>np.median(y_train_all_0, 0)
        
    elif 'mnist' in name:
        
        X_train_all_0, y_train_all_0, \
        X_test_0, y_test_0 = get_mnist()
        feat_names = None
        
        if name == 'mnist_small':
            X_train_all = X_train_all_0[::10]
            y_train_all = y_train_all_0[::10]
            X_test = X_test_0[::10]
            y_test = y_test_0[::10]
            
        elif name == 'mnist':
            X_train_all = X_train_all_0
            y_train_all = y_train_all_0
            X_test = X_test_0
            y_test = y_test_0
    
    elif name == 'ate':
        sigma_sq = 1
#        rho_t = 5
        rho_t = 0
        rho_1 = 1
        rho_2 = 0.5
        y, W, X_0 = helper_ate.simulate(sigma_sq,
                                        rho_t,
                                        rho_1,
                                        rho_2,
                                        N = 10000)
        X = np.stack([X_0,W],1)
        feat_names = None
        X_train_all, X_test, \
        y_train_all, y_test = train_test_split(X, y, test_size = 0.1, 
                                               random_state = 42)
        
    elif name == 'twins':        
        X_train_all, y_train_all, \
        X_test, y_test, feat_names = helper_ate.get_twins()
    
    elif name == 'twins_ns':        
        X_train_all, y_train_all, \
        X_test, y_test, feat_names = helper_ate.get_twins_raw()
        
#        import ipdb;ipdb.set_trace()
    if not return_bias:
        return X_train_all, X_test, \
                y_train_all, y_test,\
                feat_names
    else:
        return X_train_all, X_test, \
                y_train_all, y_test,\
                feat_names, bool_bias
    
    
def get_setup(name,
              idx_mask,
              idx_input,
              idx_target,
              verbose = 1,
              ):

    X_train_all, X_test_0, \
    y_train_all, y_test_0,\
    feat_names = get_data(name)
    
    idx_rest = np.setdiff1d(np.arange(len(X_test_0)),idx_target)
    
    X_target = X_test_0[idx_target]
    y_target = y_test_0[idx_target]
    
    X_test = X_test_0[idx_rest]
    y_test = y_test_0[idx_rest]        
    
#    p = X_train_all.shape[1]    
#    idx_omit = np.setdiff1d(range(p))                
    
    return (X_train_all, y_train_all,
            X_test, y_test,
            X_target, y_target,
            feat_names,
#            idx_omit
            )

def _get_default_target(name):
    
    if name == 'moons':
        idx_target = np.array([0])
    elif name == 'p2012':
        idx_target = np.array([3])            
    elif name == 'p2012_all':
        idx_target = np.array([266])
    elif 'mnist' in name:
        idx_target = np.array([848])
    else:
        idx_target = None
        
    return idx_target
        
def get_attack_param(name):
        
    temp = get_data(name, return_bias = 0)
    
    p = temp[0].shape[1]
    feat_names = temp[-1]
    
    if name == 'moons':
        idx_mask = np.array([0])
        idx_target = np.array([0])
    elif name == 'p2012':
        idx_mask = np.array([0])
        idx_target = np.array([3])            
    elif name == 'p2012_all':
        idx_mask = np.array([0])
        idx_target = np.array([266])
    elif 'mnist' in name:
        idx_mask = np.array([0])
#        idx_target = np.array([848]) #1495
        idx_target = np.array([1495]) #1495
    elif name == 'ate':
        idx_mask = np.array([0])
#        idx_target = np.array([0])
        idx_target = np.array([])
        
    elif name in ['twins','twins_ns']:
        target_list = ["cigar", "drink", "wtgain", 
                       "gestat", "dmeduc", "nprevist"]
        idx_mask = np.array([feat_names.index(target_list[3]),
                             feat_names.index(target_list[2])])
        idx_target = np.array([],int)
    
    idx_input = np.arange(p)
    idx_input = np.append(idx_input,[p])
    
    omit_prop = [1, 0.75, 0.5, 0.25][2]
    
    bool_partial_read = True
    if omit_prop == 1:
        bool_partial_read = False
    #%%
    return (idx_mask,
            idx_input,
            omit_prop,
            idx_target,
            bool_partial_read
            )
    
def get_data_param():
    
    name,family = [('moons','clsf'),
                   ('p2012','clsf'),
                   ('p2012_all','clsf'),
                   ('mnist','clsf'),             
                   ('mnist_small','clsf'),
                   ('ate','reg'),
                   ('twins','clsf'),
                   ('twins_ns','clsf')
                ][-1]
    
    return name,family

def get_blamm_param(name, family, **kwargs):
    
    if name == 'moons':
        return _get_blamm_param_moons(name, family)
    elif 'mnist' in name:
        return _get_blamm_param_mnist(name, family)
    elif name == 'ate':
        return _get_blamm_param_ate(name, family)
    elif name == 'twins':
        return _get_blamm_param_twins(name, family)
    elif name == 'twins_ns':
        return _get_blamm_param_twins_ns(name, family,
                                         **kwargs)
    
def _get_blamm_param_moons(name, family):
    
    #import ipdb;ipdb.set_trace()
    seed_model = 3
    lr = 0.0175
    
    epochs = 200
    loss_type = 0
    reg_lmbda = 2
        
    name_optimizer = ['adam','adagrad','SGD'][0]
    bool_scale_all = 0
    
    bool_scale_all = 0
    type_modeler = ['impute','cca','mice'][0]
    
    if 'p2012' not in name:
        model_cfg = 2
        kwargs_solver = dict(
                             solver_epochs=300,
#                             solver_epochs=10,
    #                         solver_layers=np.array([10,10]),
                             solver_layers=np.array([10]),
#                             solver_layers=[],
                             solver_seed = 42,
    #                         solver_seed = None,
#                             solver_lr = 5e-2,
#                             solver_lr = 1e-2,
#                             solver_lr = 1e-1,
                             solver_lr = 1e-1,
#                             solver_lr = 0,
#                             solver_lr = 1e-3,
                             solver_optimizer = 'adam',
#                             solver_optimizer = 'sgd',
#                             solver_warm_start = True,
    #                         solver_lambda=1e-3
                             )
    else:
        model_cfg = 6
        kwargs_solver = dict(
                             solver_epochs=1,
#                             solver_layers=np.array([64]),
                             solver_layers=np.array([32]),
                             solver_seed = 42,
    #                         solver_seed = None,
                             solver_lr = 5e-4,
#                             solver_lr = 1e-1,
    #                         solver_early = True,
                             solver_arch='rnn_p2012_all',
                             solver_optimizer = 'sgd',
    #                         solver_inshape=(48,10)
    #                         solver_activations=['tanh']
                             solver_warm_start = True,
    #                         solver_lambda=1e-3
                             )

    
    kwargs_optimizer = dict(
#            momentum=0.5
#            clipnorm=1,
#            epsilon=1e-6,
#            amsgrad=True
#            clipnorm=.75,
                            )
    print(kwargs_optimizer)
#    kwargs_optimizer = {}
    
    method = ['rev','unroll','pen','inv'][-1]
    
    if method == 'pen':
        kwargs_solver['solver_optimizer'] = name_optimizer.lower()
        kwargs_solver['solver_optimizer_kwargs'] = kwargs_optimizer
        
#    if method == 'inv'
    
    blamm_param = dict(seed_model=seed_model,
                       lr=lr,
                       epochs = epochs,
                       reg_lmbda = reg_lmbda,
                       name_optimizer = name_optimizer,
                       bool_scale_all = bool_scale_all,
                       model_cfg = model_cfg,
                       type_modeler=type_modeler,
                       kwargs_solver=kwargs_solver,
                       shuffle=False
#                       kwargs_optimizer=kwargs_optimizer
                       )
    
    if kwargs_optimizer != {}:
        blamm_param['kwargs_optimizer'] = kwargs_optimizer
    
    if method != 'inv':
        blamm_param['method'] = method
        blamm_param['n_return_states'] = 20
        
    if name == 'p2012_all':
        blamm_param['batch_size'] = 400
        
    if loss_type != 0:
        blamm_param['loss_type'] = loss_type
        
    return blamm_param

def get_mnist(class_0 = 3,
              class_1 = 8):
    
    # 1. Load the MNIST dataset
    with open(os.path.join(os.environ['LOCAL_PATH'],
                           'NotOneDrive',
                           'data',
                           'mnist.p'), "rb") as f:
    	load = pickle.load(f)
    
    y_train_all_0 = load['y_train'].astype(np.int32)
    X_train_all_0 = load['X_train'].reshape(len(y_train_all_0),-1)
    
    y_test_0 = load['y_test'].astype(np.int32)
    X_test_0 = load['X_test'].reshape(len(y_test_0),-1)
    
    #X_train_all_0 = scaler.transform(X_train_all_0)
    #print(scaler.transform([[2, 2]]))
    #%%
    
    # Create a mask to select only samples belonging to the chosen classes
    X_train_all_1 = X_train_all_0[(y_train_all_0 == class_0) | (y_train_all_0 == class_1)]
    y_train_all = y_train_all_0[(y_train_all_0 == class_0) | (y_train_all_0 == class_1)]
        
    X_test_1 = X_test_0[(y_test_0 == class_0) | (y_test_0 == class_1)]
    y_test = y_test_0[(y_test_0 == class_0) | (y_test_0 == class_1)]
        
    bool_train_all_0 = y_train_all == class_0
    bool_train_all_1 = y_train_all == class_1
    y_train_all[bool_train_all_0] = 0
    y_train_all[bool_train_all_1] = 1
        
    bool_test_0 = y_test == class_0
    bool_test_1 = y_test == class_1    
    y_test[bool_test_0] = 0
    y_test[bool_test_1] = 1
    
#    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train_all_1)
    X_train_all = scaler.transform(X_train_all_1)
    X_test = scaler.transform(X_test_1)
    
    return X_train_all, y_train_all, X_test, y_test

def _get_blamm_param_mnist(name, family):
    
    #import ipdb;ipdb.set_trace()
    seed_model = 2
    lr = 0.05
    epochs = 300
    loss_type = 3    
    reg_lmbda = 5e-3
#    reg_lmbda = 20
#    reg_lmbda = 1e-1
    
#    lr = .05
#    epochs = 100    
#    reg_lmbda = 0
        
    name_optimizer = ['adam','adagrad','SGD'][0]
    bool_scale_all = 1
        
#    model_cfg = -2
    
    bool_scale_all = 0
    type_modeler = ['impute','cca','mice'][1]
    
    if 'p2012' not in name:
        model_cfg = -2
#        model_cfg = -5
#        model_cfg = 4
        kwargs_solver = dict(
                             solver_epochs=300,
#                             solver_epochs=10,
    #                         solver_layers=np.array([10,10]),
#                             solver_layers=np.array([10]),
                             solver_layers=[],
                             solver_seed = 42,
    #                         solver_seed = None,
#                             solver_lr = 5e-2,
                             solver_lr = 1e-2,
#                             solver_lr = 1e-1,
#                             solver_lr = 1e-1,
#                             solver_lr = 0,
#                             solver_lr = 1e-3,
                             solver_optimizer = 'adam',
#                             solver_optimizer = 'sgd',
#                             solver_warm_start = True,
    #                         solver_lambda=1e-3
                             )
    else:
        model_cfg = 6
        kwargs_solver = dict(
                             solver_epochs=1,
#                             solver_layers=np.array([64]),
                             solver_layers=np.array([32]),
                             solver_seed = 42,
    #                         solver_seed = None,
                             solver_lr = 5e-4,
#                             solver_lr = 1e-1,
    #                         solver_early = True,
                             solver_arch='rnn_p2012_all',
                             solver_optimizer = 'sgd',
    #                         solver_inshape=(48,10)
    #                         solver_activations=['tanh']
                             solver_warm_start = True,
    #                         solver_lambda=1e-3
                             )

    
    kwargs_optimizer = dict(
#            momentum=0.5
#            clipnorm=1,
#            epsilon=1e-6,
#            amsgrad=True
#            clipnorm=.75,
                            )
    print(kwargs_optimizer)
#    kwargs_optimizer = {}
    
    method = ['rev','unroll','pen','inv'][-1]
    
    if method == 'pen':
        kwargs_solver['solver_optimizer'] = name_optimizer.lower()
        kwargs_solver['solver_optimizer_kwargs'] = kwargs_optimizer
        
#    if method == 'inv'
    
    blamm_param = dict(seed_model=seed_model,
                       lr=lr,
                       epochs = epochs,
                       reg_lmbda = reg_lmbda,
                       name_optimizer = name_optimizer,
                       bool_scale_all = bool_scale_all,
                       model_cfg = model_cfg,
                       type_modeler=type_modeler,
                       kwargs_solver=kwargs_solver,
                       shuffle=False
#                       kwargs_optimizer=kwargs_optimizer
                       )
    
    if kwargs_optimizer != {}:
        blamm_param['kwargs_optimizer'] = kwargs_optimizer
    
    if method != 'inv':
        blamm_param['method'] = method
        blamm_param['n_return_states'] = 20
        
    if name == 'p2012_all':
        blamm_param['batch_size'] = 400
        
    if loss_type != 0:
        blamm_param['loss_type'] = loss_type
        
    return blamm_param

def _get_blamm_param_ate(name, family):
    
    #import ipdb;ipdb.set_trace()
    seed_model = 2
#    seed_model = 3
#    lr = 5e-3
#    lr = 0.01
    lr = 0.05
#    lr = 1e-3
#    lr = 0.001
#    lr = 0.
#    lr =    0.5
#    epochs = 1000
    epochs = 300
#    epochs = 600
#    reg_lmbda = 2
#    loss_type = 2
    loss_type = 5
    reg_lmbda = 0.5
    
#    lr = 0
#    reg_lmbda = 1e-2
#    reg_lmbda = 20
#    reg_lmbda = 1e-1
    
#    lr = .05
#    epochs = 100    
#    reg_lmbda = 0
        
    name_optimizer = ['adam','adagrad','SGD'][0]
    bool_scale_all = 1
        
#    model_cfg = -2
    
    bool_scale_all = 0
    type_modeler = ['impute','cca','mice'][0]
    
    if 'p2012' not in name:
        model_cfg = -2
#        model_cfg = -5
#        model_cfg = 4
        kwargs_solver = dict(
                             solver_epochs=40,
#                             solver_epochs=10,
    #                         solver_layers=np.array([10,10]),
#                             solver_layers=np.array([10]),
                             solver_layers=[],
                             solver_seed = 42,
    #                         solver_seed = None,
#                             solver_lr = 5e-2,
#                             solver_lr = 1e-2,
                             solver_lr = 1e-1,
#                             solver_lr = 1e-1,
#                             solver_lr = 0,
#                             solver_lr = 1e-3,
                             solver_optimizer = 'adam',
#                             solver_optimizer = 'sgd',
#                             solver_warm_start = True,
    #                         solver_lambda=1e-3
                             )
    
    kwargs_optimizer = dict(
#            momentum=0.5
#            clipnorm=1,
#            epsilon=1e-6,
#            amsgrad=True
#            clipnorm=.75,
                            )
    print(kwargs_optimizer)
#    kwargs_optimizer = {}
    
    method = ['rev','unroll','pen','inv'][-1]
    
    if method == 'pen':
        kwargs_solver['solver_optimizer'] = name_optimizer.lower()
        kwargs_solver['solver_optimizer_kwargs'] = kwargs_optimizer
    
    blamm_param = dict(seed_model=seed_model,
                       lr=lr,
                       epochs = epochs,
                       reg_lmbda = reg_lmbda,
                       name_optimizer = name_optimizer,
                       bool_scale_all = bool_scale_all,
                       model_cfg = model_cfg,
                       type_modeler=type_modeler,
                       kwargs_solver=kwargs_solver,
                       shuffle=False
#                       kwargs_optimizer=kwargs_optimizer
                       )
    
    if kwargs_optimizer != {}:
        blamm_param['kwargs_optimizer'] = kwargs_optimizer
    
    if method != 'inv':
        blamm_param['method'] = method
        blamm_param['n_return_states'] = 20
                
    if loss_type != 0:
        blamm_param['loss_type'] = loss_type
        
    return blamm_param

def _get_blamm_param_twins(name, family):
    
    #import ipdb;ipdb.set_trace()
    seed_model = 2
#    seed_model = 3
#    lr = 5e-3
#    lr = 0.01
    lr = 0.05
#    lr = 1e-3
#    lr = 0.001
#    lr = 0.
#    lr =    0.5
#    epochs = 1000
    epochs = 300
#    epochs = 600
#    reg_lmbda = 1
#    loss_type = 2
    loss_type = 6
    reg_lmbda = 0.5
    
#    lr = 0
#    reg_lmbda = 1e-2
#    reg_lmbda = 20
#    reg_lmbda = 1e-1
    
#    lr = .05
#    epochs = 100    
#    reg_lmbda = 0
        
    name_optimizer = ['adam','adagrad','SGD'][0]
    bool_scale_all = 1
        
#    model_cfg = -2
    
    bool_scale_all = 0
    type_modeler = ['impute','cca','mice'][0]
    
    if 'p2012' not in name:
        model_cfg = -2
#        model_cfg = -5
#        model_cfg = 4
        kwargs_solver = dict(
                             solver_epochs=40,
#                             solver_epochs=10,
    #                         solver_layers=np.array([10,10]),
#                             solver_layers=np.array([10]),
                             solver_layers=[],
                             solver_seed = 42,
    #                         solver_seed = None,
#                             solver_lr = 5e-2,
#                             solver_lr = 1e-2,
                             solver_lr = 1e-1,
#                             solver_lr = 1e-1,
#                             solver_lr = 0,
#                             solver_lr = 1e-3,
                             solver_optimizer = 'adam',
#                             solver_optimizer = 'sgd',
#                             solver_warm_start = True,
    #                         solver_lambda=1e-3
                             )
    
    kwargs_optimizer = dict(
#            momentum=0.5
#            clipnorm=1,
#            epsilon=1e-6,
#            amsgrad=True
#            clipnorm=.75,
                            )
    print(kwargs_optimizer)
#    kwargs_optimizer = {}
    
    method = ['rev','unroll','pen','inv'][-1]
    
    if method == 'pen':
        kwargs_solver['solver_optimizer'] = name_optimizer.lower()
        kwargs_solver['solver_optimizer_kwargs'] = kwargs_optimizer
    
    blamm_param = dict(seed_model=seed_model,
                       lr=lr,
                       epochs = epochs,
                       reg_lmbda = reg_lmbda,
                       name_optimizer = name_optimizer,
                       bool_scale_all = bool_scale_all,
                       model_cfg = model_cfg,
                       type_modeler=type_modeler,
                       kwargs_solver=kwargs_solver,
                       shuffle=False,
#                       type_scaler = 'minmax'
#                       kwargs_optimizer=kwargs_optimizer
                       )
    
    if kwargs_optimizer != {}:
        blamm_param['kwargs_optimizer'] = kwargs_optimizer
    
    if method != 'inv':
        blamm_param['method'] = method
        blamm_param['n_return_states'] = 20
                
    if loss_type != 0:
        blamm_param['loss_type'] = loss_type
        
    return blamm_param

def _get_blamm_param_twins_ns(name, family, 
                              omit_prop):
    
    #import ipdb;ipdb.set_trace()
    seed_model = 2
    lr = 0.05
    epochs = 300
    loss_type = 6
    
    if omit_prop == 1:
        reg_lmbda = 5e-1
        bool_resolve = False
    elif omit_prop == .75:
        reg_lmbda = 2.5e-1
        bool_resolve = True
    elif omit_prop in [.5,.25]:
        reg_lmbda = 1e-4
        bool_resolve = True
            
    name_optimizer = ['adam','adagrad','SGD'][0]
    
    bool_scale_all = 0
    type_modeler = ['impute','cca','mice'][0]
    
    model_cfg = -2    
    kwargs_solver = dict(
                         solver_layers=[],
                         )
        
    kwargs_optimizer = dict()
    print(kwargs_optimizer)
    
    method = ['rev','unroll','pen','inv'][-1]
    
    if method == 'pen':
        kwargs_solver['solver_optimizer'] = name_optimizer.lower()
        kwargs_solver['solver_optimizer_kwargs'] = kwargs_optimizer
    
    blamm_param = dict(seed_model=seed_model,
                       lr=lr,
                       epochs = epochs,
                       reg_lmbda = reg_lmbda,
                       name_optimizer = name_optimizer,
                       bool_scale_all = bool_scale_all,
                       model_cfg = model_cfg,
                       type_modeler=type_modeler,
                       kwargs_solver=kwargs_solver,
                       shuffle=False,
                       type_scaler = 'minmax'
#                       kwargs_optimizer=kwargs_optimizer
                       )
    
    if kwargs_optimizer != {}:
        blamm_param['kwargs_optimizer'] = kwargs_optimizer
    
    if method != 'inv':
        blamm_param['method'] = method
        blamm_param['n_return_states'] = 20
                
    if loss_type != 0:
        blamm_param['loss_type'] = loss_type
    
    if not bool_resolve:
        blamm_param['bool_resolve'] = bool_resolve
        
    return blamm_param