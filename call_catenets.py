# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np

from helper_catenets import _estimate_and_score_ate

parser = argparse.ArgumentParser()
parser.add_argument('savePath',
                    type=str)
parser.add_argument('family',
                    type=str)
parser.add_argument('type_predictor',
                    type=str)
args = parser.parse_args()
    
if __name__ == '__main__':
    import sys
    print(sys.version)
         
    with open(os.path.join(args.savePath,'X_train.npy'), 'rb') as f:
        X_train = np.load(f)
    with open(os.path.join(args.savePath,'y_train.npy'), 'rb') as f:
        y_train = np.load(f)
        
    with open(os.path.join(args.savePath,'X_test.npy'), 'rb') as f:
        X_test = np.load(f)
    with open(os.path.join(args.savePath,'y_test.npy'), 'rb') as f:
        y_test = np.load(f)        

    scores = _estimate_and_score_ate(X_train, y_train,
                                     X_test, y_test,
                                     args.family,
                                     args.type_predictor)
    scores = np.array(scores)

    with open(os.path.join(args.savePath,'temp.npyz'), 'wb') as f:
        np.savez(f, scores=scores)