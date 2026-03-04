# -*- coding: utf-8 -*-
import pandas as pd
import os
import copy
import pickle 

import inspect
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import normalize,scale
try:
    from folktables import ACSDataSource, ACSIncome
except:
    print('folktables not ins')

from ucimlrepo import fetch_ucirepo 

#%%
filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))

def get_adult_v0():
        
    df_0 = pd.read_csv(os.path.join(currPath,'adult.data'), 
                       sep=",", 
                       header = None
                       )
    
    cols_0 = ['age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'class']
    
    df_0.columns = cols_0
    
    df = df_0.iloc[:,:-1]
    cols = cols_0[:-1]
    
    y_0 = df_0.iloc[:,-1:].to_numpy()
    enc_y = OrdinalEncoder()
    y = enc_y.fit_transform(y_0)[:,0]
    #%%
    cols_c = ['age',        
            'fnlwgt',
            'capital-gain',
            'capital-loss',
            'hours-per-week']
    
    #df_sub = df[con]
    idx_c = np.array([cols.index(i) for i in cols_c])
    idx_d = np.setdiff1d(range(len(cols)),idx_c)
    
    X_d_0 = df.iloc[:,idx_d].to_numpy()
    
    #enc = OrdinalEncoder()
    #X_d_1 = enc.fit_transform(X_d_0)
    #Rs = [len(cat) for cat in enc.categories_]
    
    #print(np.sum(Rs)+len(idx_c)*np.array([29,13,94]))
    
    enc2 = OneHotEncoder(sparse = False)
    #X_d = enc2.fit_transform(X_d_1)
    X_d = enc2.fit_transform(X_d_0)
    
    cols_d_0 = [cols[i] for i in idx_d]
    cols_d = enc2.get_feature_names_out(input_features=cols_d_0)
    
    #%%
    X_c = df.iloc[:,idx_c].to_numpy()
    
    #%%
    
    if 0:
        bool_filter = X_d.sum(0)>100
        X_d = X_d[:,bool_filter]
        cols_d = cols_d[bool_filter]
    
    X = np.concatenate([X_c,X_d],1)
    cols_enc = cols_c + cols_d.tolist()
        
    return X, y, cols_enc

def get_enc(X_0, cols, cols_c,
            bool_stdrz = False,
            bool_norm = 0,
            drop = None):
    
    idx_c = np.array([cols.index(i) for i in cols_c])
    idx_d = np.setdiff1d(range(len(cols)),idx_c)
    
    #Select and encodes the discreet columns
    if type(X_0) == np.ndarray:
        X_d_0 = X_0[:,idx_d]
    else:
        X_d_0 = X_0.iloc[:,idx_d].to_numpy()
    
    enc2 = OneHotEncoder(sparse = False, drop = drop)
    X_d_1 = enc2.fit_transform(X_d_0)
    
    cols_d_0 = [cols[i] for i in idx_d]
    cols_d = enc2.get_feature_names_out(input_features=cols_d_0)
    
    #Select the continuous columns
    if type(X_0) == np.ndarray:
        X_c_0 = X_0[:,idx_c]
    else:
        X_c_0 = X_0.iloc[:,idx_c].to_numpy(float)   
        
    cols_enc = cols_c + cols_d.tolist()
    
#    import ipdb;ipdb.set_trace()
    
    if bool_stdrz:
        print('Standard')
        X_c = scale(X_c_0)
    else:
        X_c = X_c_0
    
#    bool_norm = 0
    
    if bool_norm:
        X_d = normalize(X_d_1)
    else:
        X_d = X_d_1
        
    X = np.concatenate([X_c,X_d], 1)
    
    bool_c = np.zeros(X.shape[1])
    bool_c[:X_c.shape[1]] = 1
    
    return X, cols_enc, bool_c

def get_acs(bool_stdrz=0,
            bool_norm=0,
            drop = None,
            bool_only_d = False): 
    
    '''
    Follows the examples in
    https://github.com/socialfoundations/folktables/tree/main
    '''
    
    data_source = ACSDataSource(survey_year='2018', 
                                horizon='1-Year', 
                                survey='person')
    
    acs_data = data_source.get_data(states=["NY",'CA'], download=True)
    X_0, y, group = ACSIncome.df_to_numpy(acs_data)
    
    cols_0 = ACSIncome.features    
        
    cols_c =[
            'AGEP',    
            'WKHP',
            ]
    
    X, cols, bool_c = get_enc(X_0, cols_0, cols_c, 
                              bool_stdrz=bool_stdrz,
                              bool_norm=bool_norm,
                              drop = drop)    
    
    if bool_only_d:
#        import ipdb;ipdb.set_trace()
        n_c = int(bool_c.sum())
        X = X[:,n_c:]
        cols = cols[n_c:]
        
    return X, y, cols

def get_adult(bool_stdrz=0,
              bool_norm=0,
              bool_only_c = 0,
              drop = None,
              return_bool_c = False):
        
    df_0 = pd.read_csv(os.path.join(currPath,'adult.data'), 
                       sep=",", 
                       header = None
                       )
    
    cols_0 = ['age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'class']
    
    df_0.columns = cols_0
    
    df = df_0.iloc[:,:-1]
    cols_1 = cols_0[:-1]
    
    y_0 = df_0.iloc[:,-1:].to_numpy()
    enc_y = OrdinalEncoder()
    y = enc_y.fit_transform(y_0)[:,0]
    #%%
    cols_c = ['age',        
            'fnlwgt',
            'capital-gain',
            'capital-loss',
            'hours-per-week']
    
    X_0 = df
    
    X, cols, bool_c = get_enc(X_0, cols_1, cols_c, 
                              bool_stdrz=bool_stdrz,
                              bool_norm=bool_norm,
                              drop = drop)
    
    if bool_only_c:
#        import ipdb;ipdb.set_trace()
        n_c = int(bool_c.sum())
        X = X[:,:n_c]
        cols = cols[:n_c]
    
    if not return_bool_c:
        return X, y, cols
    else:
        return X, y, cols, bool_c

def get_german(bool_stdrz=0,
               bool_norm=0,
               drop = None,
               bool_only_d = 0,
               bool_fetch = False):
        
    if bool_fetch:
        # fetch dataset 
        statlog_german_credit_data = fetch_ucirepo(id=144) 
          
        # data (as pandas dataframes) 
        X_0 = statlog_german_credit_data.data.features 
        y_0 = statlog_german_credit_data.data.targets 
    else:
        with open(os.path.join(currPath,
                               'statlog_german_credit_data.p'), "rb") as f:
            load = pickle.load(f)
        X_0 = load['features']
        y_0 = load['targets']
      
    # metadata 
#    print(statlog_german_credit_data.metadata) 
      
    # variable information 
#    print(statlog_german_credit_data.variables) 
    
    if bool_fetch:
        temp = statlog_german_credit_data.variables['type'].to_numpy()
    else:
        temp = load['variables']['type'].to_numpy()
    bool_d = (temp == 'Categorical') | (temp == 'Binary')
    
    #%%
    if bool_fetch:
        cols_0 = statlog_german_credit_data.variables.iloc[:-1,0].tolist()
    else:
        cols_0 = load['variables'].iloc[:-1,0].tolist()
    cols_c = [cols_0[i] for i in np.where(~bool_d)[0]]
    
    y = y_0.to_numpy()[:,0]-1
    
    
    X, cols, bool_c = get_enc(X_0, cols_0, cols_c, 
                              bool_stdrz=bool_stdrz,
                              bool_norm=bool_norm,
                              drop = drop
                              )    
    
    if bool_only_d:
#        import ipdb;ipdb.set_trace()
        n_c = int(bool_c.sum())
        X = X[:,n_c:]
        cols = cols[n_c:]
    
#    import ipdb;ipdb.set_trace()
    
    return X, y, cols

def get_wq(bool_stdrz=0,
           bool_norm=0,
           bool_color = False,
           bool_fetch = False): 
  
    if bool_fetch:
        # fetch dataset 
        wine_quality = fetch_ucirepo(id=186)
          
        # data (as pandas dataframes) 
        X_0 = wine_quality.data.features
        y_0 = wine_quality.data.targets
    else:
        with open(os.path.join(currPath,'wine_quality.p'), "rb") as f:
            load = pickle.load(f)
        X_0 = load['features']
        y_0 = load['targets']
    
    # metadata 
#    print(wine_quality.metadata) 
      
    # variable information 
#    print(wine_quality.variables) 
    cols_0 = X_0.columns.tolist()
    
    if not bool_color:    
        cols = cols_0    
        X_1 = X_0.to_numpy()
        
        if bool_stdrz:
            print('Standard')
            X = scale(X_1)
        else:
            X = X_1
    else:
        X_1 = wine_quality.data.original['color']
        X_2 = pd.concat([X_0,X_1],axis=1)
        cols_1 = cols_0 + ['color']
        
        cols_c = cols_0
        X, cols, bool_c = get_enc(X_2, cols_1, cols_c, 
                              bool_stdrz=bool_stdrz,
                              bool_norm=bool_norm,
                              drop = 'first'
                              )    
        
    y = (y_0.to_numpy()[:,0]>5).astype(int)
    
    return X, y, cols

def get_ab(bool_stdrz=0,
           bool_norm=0):
     
    # fetch dataset 
    abalone = fetch_ucirepo(id=1) 
      
    # data (as pandas dataframes) 
    X_0 = abalone.data.features 
    y_0 = abalone.data.targets 
    
    cols_0 = X_0.columns.tolist()
    temp_df = abalone.variables.set_index('name')
    temp = temp_df.loc[cols_0]['type'].to_numpy()
    
    bool_d = (temp == 'Categorical') | (temp == 'Binary')
    
    cols_c = [cols_0[i] for i in np.where(~bool_d)[0]]
    
    y = y_0.to_numpy()[:,0]-1    
    
    X, cols, bool_c = get_enc(X_0, cols_0, cols_c, 
                              bool_stdrz=bool_stdrz,
                              bool_norm=bool_norm
                              )
    
#    import ipdb;ipdb.set_trace()
    return X, y, cols

def get_diabetes(bool_stdrz=0,
                 bool_norm=0):
    
    url = 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'

    df = pd.read_table(url,
                       delimiter='\t')
    
    X_0 = df.iloc[:,:-1]
    cols_0 = X_0.columns.tolist()
    cols_c = copy.deepcopy(cols_0)
    cols_c.remove('SEX')
    
    y = df.iloc[:,-1].to_numpy()
    
    X, cols, bool_c = get_enc(X_0, cols_0, cols_c, 
                              bool_stdrz=bool_stdrz,
                              bool_norm=bool_norm,
                              drop='first'
                              )
    
    return X, y, cols