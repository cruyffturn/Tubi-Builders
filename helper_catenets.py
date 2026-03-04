'''
Adapted from CATENets-main/experiments/experiments_inductivebias_NeurIPS21
'''
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from catenets.models.jax import (
    DRNET_NAME,
    FLEXTE_NAME,
    OFFSET_NAME,
    T_NAME,
    TARNET_NAME,
    DRNet,
    FlexTENet,
    OffsetNet,
    TARNet,
    TNet,
)

RESULT_DIR = Path("results/experiments_inductive_bias/twins")
SEP = "_"

PARAMS_DEPTH = {"n_layers_r": 1, "n_layers_out": 1}
PARAMS_DEPTH_2 = {
    "n_layers_r": 1,
    "n_layers_out": 1,
    "n_layers_r_t": 1,
    "n_layers_out_t": 1,
}
PENALTY_DIFF = 0.01
PENALTY_ORTHOGONAL = 0.1

def _estimate_and_score_ate(X_train_0, y_train,
                           X_test, y_test,
                           family,
                           type_predictor):
    
    if type_predictor == 'tnet':
        model = TNet(**PARAMS_DEPTH)
        bool_mu = True
        
    elif type_predictor == 'tarnet':
        model = TARNet(**PARAMS_DEPTH)
        bool_mu = False
        
    if family == 'clsf':
        model.set_params(**{"binary_y": True})
    
    
    model.fit(X=X_train_0[:,:-1], y=y_train, w=X_train_0[:,-1])
    print("Model training complete.")

    if bool_mu:
        cate_pred_out, mu_0, mu_1 = model.predict(X_test[:,:-1], 
                                                  return_po=True)
        
        cate_pred_out = np.array(cate_pred_out)
        mu_0 = np.array(mu_0)
        mu_1 = np.array(mu_1)
    else:
        cate_pred_out = model.predict(X_test[:,:-1])
        mu_0 = None
        mu_1 = None
        
#        import ipdb;ipdb.set_trace()
#    ate, ate_0, ate_1 = _score_ate(X_test, model, family)
    ate = cate_pred_out.mean()
    if bool_mu:
        ate_0 = mu_0.mean()
        ate_1 = mu_1.mean()
        
        auc_0 = roc_auc_score(y_test[:,0], mu_0)
        auc_1 = roc_auc_score(y_test[:,1], mu_1)
    else:
        ate_0 = np.nan
        ate_1 = np.nan
        
        auc_0 = np.nan
        auc_1 = np.nan
        
    
    return ate, ate_0, ate_1, auc_0, auc_1