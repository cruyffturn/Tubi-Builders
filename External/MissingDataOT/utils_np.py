"""
This code is derived from https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py
"""

import numpy as np
from scipy import optimize
from scipy.special import expit

def pick_coeffs_np(X, idxs_obs=None, idxs_nas=None, self_mask=False):
#    import pdb;pdb.set_trace()
    n, d = X.shape
    if self_mask:
        coeffs = np.random.randn(d)
        Wx = X * coeffs
        coeffs /= np.std(Wx, axis=0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.randn(d_obs, d_na)
        Wx = X[:, idxs_obs] @ coeffs
        coeffs /= np.std(Wx, axis=0)
    return coeffs

def fit_intercepts_np(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = np.zeros(d)
        for j in range(d):
            def f(x):
                return expit(X * coeffs[j] + x).mean() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return expit(X.dot(coeffs[:, j]) + x).mean() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    
    return intercepts

def _MNAR_self_mask_logistic(X, p, seed_1):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    
    
    """

    n, d = X.shape
    np.random.seed(seed_1)
    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs_np(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts_np(X, coeffs, p, self_mask=True)
    
    return coeffs, intercepts

def _MNAR_mask_logistic(X, p, seed_1, idxs_nas):
    """
    weights are random and the intercept is selected to attain the desired proportion of missing values.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)
    Returns
    -------
    """
    n, d = X.shape
    
    idxs_params = np.arange(d)
    
    np.random.seed(seed_1)
    ### The parameters of this logistic model are random.
    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs_np(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts_np(X[:, idxs_params], coeffs, p)

    return coeffs, intercepts