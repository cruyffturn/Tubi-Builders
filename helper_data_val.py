# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
'''
How the metrics are input?

'''
import os
import inspect

import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

try:
    from opendataval.dataloader import DataFetcher
    from opendataval.dataloader import Register
    from opendataval.model.api import ClassifierSkLearnWrapper, RegressionSkLearnWrapper
    from opendataval.experiment import ExperimentMediator
    
    from opendataval.dataval import (
        AME,
        DVRL,
        BetaShapley,
        DataBanzhaf,
        DataOob,
        DataShapley,
        InfluenceSubsample,
        KNNShapley,
        LavaEvaluator,
        LeaveOneOut,
        RandomEvaluator,
        RobustVolumeShapley,
    )
    
    from opendataval.experiment.exper_methods import (
        save_dataval
    )
except:
    pass
filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
sharedPath = os.path.join(os.environ['SHARED'],
                              'Code',
                              currPath.split(os.environ['CODE_PATH'])[1][1:])

import shlex
import subprocess

def get_X_y_concat(X_train, y_train,
                   X_val, y_val):
    
    X = np.concatenate([X_train,X_val],0)
    y = np.concatenate([y_train,y_val],0)
    
    return X, y

def get_data_val(X_0, y_0,                  
                 bool_classify,
                 n_train,
                 type_data_val,         
                 seed = 42,
                 bool_std = True):
        
    '''
    In:
        X:          N,p
        y:          N,
        
    Out:
        data_val:   N,
    '''
#    if type_data_val == 'knn':
#        bool_std = True
#    elif type_data_val == 'lava':
#        bool_std = False
        
    cachePath = os.path.join(sharedPath,'cache')
    dataset_name = "dataset_%.3f"%(np.random.rand(1)[0])
    
#    if bool_classify:        
#        y = OneHotEncoder(sparse_output=False).fit_transform(y_0[:,np.newaxis])
#    else:
#        y = y_0
    y = y_0
    if bool_std:
        print('scaling')
        X = StandardScaler().fit_transform(X_0)
    else:
        X = X_0
        
    # Register a dataset from arrays X and y
    pd_dataset = Register(dataset_name=dataset_name,
                          one_hot=bool(bool_classify)).from_data(X, y)

    #cache_dir = 
    fetcher = (
        DataFetcher(dataset_name, cachePath, False,random_state=seed)
        .split_dataset_by_indices([i for i in range(n_train)],
                                   [i for i in range(n_train,len(X))],
                                   [])
    )
        
    if bool_classify:
        metric_name = 'accuracy'
        pred_model = ClassifierSkLearnWrapper(LogisticRegression, 
                                              fetcher.label_dim[0])
    else:
        metric_name = 'neg_mse'
        pred_model = RegressionSkLearnWrapper(LinearRegression)
    
    exper_med = ExperimentMediator(fetcher, 
                                   pred_model,
                                   metric_name=metric_name)
        
    if type_data_val == 'random':
        data_evaluators = [RandomEvaluator(random_state=seed)]
        
    elif type_data_val == 'loo':
        data_evaluators = [LeaveOneOut(random_state=seed)]
    elif type_data_val == 'loo':
        data_evaluators = [LeaveOneOut(random_state=seed)]
    elif type_data_val == 'lava':
        data_evaluators = [LavaEvaluator(random_state=seed)]
    elif type_data_val == 'knn':
        data_evaluators = [KNNShapley(k_neighbors=len(X)-n_train)]
        
#    import pdb;pdb.set_trace()
    exper_med = exper_med.compute_data_values(data_evaluators=data_evaluators)
    
    results = exper_med.evaluate(save_dataval)
    
    if np.any(results['indices'].to_numpy(int) != np.arange(n_train)):
        raise ValueError
    
    data_values = results['data_values'].to_numpy(float)
    
    return data_values

def get_data_value_sanity(X, y,
                          type_data_val,
                          seed = 42):
    '''
    In:
        X:          N,p
        y:          N,
        
    Out:
        data_val:   N,
    '''
    if type_data_val == 'random':
        np.random.seed(seed)
        
        data_val = np.random.choice(np.arange(len(X)),
                                    size=len(X),
                                    replace=False)
        
        
    return data_val


def get_data_val_wrapped(X_all, y_all,                  
                         bool_classify,
                         n_train,
                         type_data_val,
                         seed):
        
    miniconda = 'miniconda3/envs/lava/bin/python'
    script_path = os.path.join(currPath,'call_lava.py')
    savePath = os.path.join(sharedPath,'temp_data_val')
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    file_x = os.path.join(savePath,'X.npy')
    if os.path.exists(file_x):
        os.remove(file_x)
        
    with open(file_x, 'wb') as f:
        np.save(f, X_all)
    
    file_y = os.path.join(savePath,'y.npy')
    if os.path.exists(file_y):
        os.remove(file_y)
        
    with open(file_y, 'wb') as f:
        np.save(f, y_all) 
    
    file_z = os.path.join(savePath,'data_val.npy')
    if os.path.exists(file_z):
        os.remove(file_z)    

    command = "%s %s %s %i %i %s %i"%(miniconda,
                                      script_path,
                                                 savePath,
                                                 bool_classify,
                                                 n_train,
                                                 type_data_val,
                                                 seed)
                                                 
    print(command)
    args = shlex.split(command)
    my_subprocess = subprocess.Popen(args)
    my_subprocess.wait()
    
    
    with open(file_z, 'rb') as f:
        data_val = np.load(f)
        
    return data_val

def plot_with_ci(budget_per_vec, y, ax):
    
    import seaborn as sns
    xaxis = np.repeat(budget_per_vec[:,np.newaxis],20)
    df_temp = pd.DataFrame({'x':xaxis, 'y':y})
    sns.lineplot(data=df_temp, x='x', y='y', ax=ax)
    