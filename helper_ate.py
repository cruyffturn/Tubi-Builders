import os
import numpy as np
import pandas as pd
import inspect
import matplotlib.pyplot as plt

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
from scipy.special import expit
from External.CATENets_custom_2 import dataset_twins
from sklearn.model_selection import train_test_split
#%%

def simulate(sigma_sq,
            rho_t,
            rho_1,
            rho_2,
            N = 1000):
    
    np.random.seed(42)
    
    Z = np.random.normal(0, np.sqrt(sigma_sq),[N,2])
    X = Z[:,0]
    W = np.random.binomial(1, expit(X*rho_t))
    
    Y = X*rho_1 +W*rho_2 + Z[:,1]
    
    return Y, W, X

def prepare_twins(outfile):

    '''
    Following https://github.com/AliciaCurth/CATENets/blob/main/experiments/experiments_benchmarks_NeurIPS21/twins_experiments_catenets.py
    '''
    
    feat_list = [
        "dmage",
        "mpcb",
        "cigar",
        "drink",
        "wtgain",
        "gestat",
        "dmeduc",
        "nprevist",
        "dmar",
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "dtotord",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
        "adequacy_1",
        "adequacy_2",
        "adequacy_3",
        "pldel_1",
        "pldel_2",
        "pldel_3",
        "pldel_4",
        "pldel_5",
        "resstatb_1",
        "resstatb_2",
        "resstatb_3",
        "resstatb_4",
    ]

    # use existing file
    df_train = pd.read_csv(outfile + '_train.csv')
    X_train = np.asarray(df_train[feat_list])
    y_train = np.asarray(df_train[["y"]]).reshape((-1,))
    w_train = np.asarray(df_train[["w"]]).reshape((-1,))

    df_test = pd.read_csv(outfile + '_test.csv')
    X_test = np.asarray(df_test[feat_list])
    y0_test = np.asarray(df_test[["y0"]]).reshape((-1,))
    y1_test = np.asarray(df_test[["y1"]]).reshape((-1,))
    
    return X_train, y_train, w_train, \
            X_test, y0_test, y1_test, \
            feat_list

def get_twins():
    
    outfile = os.path.join(currPath,'preprocessed_0.5_None_0.5_0')
    X_train_0, y_train, w_train_0, \
    X_test_0, y0_test, y1_test, \
    feat_names = prepare_twins(outfile)

    X_train = np.concatenate([X_train_0,w_train_0[:,np.newaxis]],1)
    X_test = np.concatenate([X_test_0,np.full([len(X_test_0),1],np.nan)],
                             1)
    y_test = np.stack([y0_test, y1_test],1)
    feat_names.append('treatment')
    return X_train, y_train, X_test, y_test, feat_names

def get_twins_raw(random_state = 0):

    X_0, w, \
    y, po_y,\
    _, _, \
    feat_names = dataset_twins.preprocess(os.path.join(currPath,'Twin_Data.csv'),
                                             train_ratio = 1,
                                             seed= 0,
                                             bool_scale = 0
                                             )
    
    X = np.concatenate([X_0, w[:,np.newaxis]], 1)
    feat_names.append('treatment')
    
    X_train, X_test,\
    y_train, _,\
    _, y_test = train_test_split(X, y, po_y,
                                 test_size=0.5, 
                                 random_state=random_state)
#    import ipdb;ipdb.set_trace()
    return X_train, y_train, X_test, y_test, feat_names