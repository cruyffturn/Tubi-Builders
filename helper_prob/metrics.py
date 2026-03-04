# -*- coding: utf-8 -*-
import numpy as np
import numpy.ma as ma


def getCorr_XY(X, Y, return_sigma = False):
    
    '''
    Option: Pearson
    
    In:
        X:  N,p
        Y:  N,p
    
    '''
    
    N = X.shape[0]
    
    mu_X = np.average( X,0)
    mu_Y = np.average( Y,0)
    
    std_X = np.std(X,0)
    std_Y = np.std(Y,0)
    
    Sig = np.matmul((X-mu_X).T, 
                    (Y-mu_Y))/N

    if not return_sigma:    
        denom = np.outer(std_X ,std_Y)
        
        pearSig = Sig / denom
        
        return pearSig
    
    else:
        return Sig
    
