#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
def get_mesh_ravel(X):
    
    '''
    In:
        X:              L;K_l
    Out:
        mesh_unroll:    (\prod_l K_l),L
    
    '''
    X_mesh = np.meshgrid(*X, 
                     copy=True, 
                     sparse=False, 
                     indexing='ij')

    X_mesh_ravel = np.stack(X_mesh).reshape(len(X_mesh),-1).T
    
    return X_mesh_ravel