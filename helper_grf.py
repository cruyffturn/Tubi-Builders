import numpy as np
try:
    from rpy2.robjects import numpy2ri
    import rpy2
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri
except:
    print('R not installed')
    
import gc
    
def _estimate_and_score_ate(X_train_0, y_train,
                            X_test_0, y_test,
                            family,
                            type_estimator):
    
    '''
    '''
    numpy2ri.activate()
    pandas2ri.activate()
    rpackages.importr('grf')
    
    seed = 42
    
    rpy2.robjects.r("""
        f <- function(X, Y, W, X.test, seed) {
            Y = as.vector(Y)
            W = as.vector(W)
            #W = factor(W, levels = c(0, 1), labels = c("0", "1"))
            #Y = factor(Y, levels = c(0, 1), labels = c("0", "1"))
            #print(typeof(W))
            #print(typeof(Y))
            #na_matrix <- is.na(X)
            #print(na_matrix)
            
            tau.forest <- causal_forest(X, Y, W, seed=seed)
            tau.hat <- predict(tau.forest, X.test)
                    
            return(tau.hat)
        }
        """)

    X_train = X_train_0[:,:-1]
    W_train = X_train_0[:,-1].astype(int)

    X_test = X_test_0[:,:-1]
    
    load = rpy2.robjects.r['f'](X_train, y_train, W_train, X_test, seed)
    cate = load.to_numpy()[:,0]
    
    ate = cate.mean()
    
    gc.collect()
    pandas2ri.deactivate()
    numpy2ri.deactivate()
    
    return ate, np.nan, np.nan, np.nan, np.nan

def _estimate_and_score_ate_dr(X_train_0, y_train,
                               X_test_0, y_test,
                               family,
                               type_estimator,
                               seed = 42):
    
    '''
    '''
    numpy2ri.activate()
    pandas2ri.activate()
    rpackages.importr('grf')
    print('using seed', seed)
#    seed = 42
    
    rpy2.robjects.r("""
        f <- function(X, Y, W, X.test, seed) {
            Y = as.vector(Y)
            W = as.vector(W)
            #W = factor(W, levels = c(0, 1), labels = c("0", "1"))
            #Y = factor(Y, levels = c(0, 1), labels = c("0", "1"))
            #print(typeof(W))
            #print(typeof(Y))
            #na_matrix <- is.na(X)
            #print(na_matrix)
            
            tau.forest <- causal_forest(X, Y, W, seed=seed)
            ate <- average_treatment_effect(tau.forest, target.sample = "all")
                    
            return(ate)
        }
        """)

    X_train = X_train_0[:,:-1]
    W_train = X_train_0[:,-1].astype(int)

    X_test = X_test_0[:,:-1]
    
#    import ipdb;ipdb.set_trace()
    load = rpy2.robjects.r['f'](X_train, y_train, W_train, X_test, seed)
    ate = load[0]
    
    gc.collect()
    pandas2ri.deactivate()
    numpy2ri.deactivate()
    
    return ate, np.nan, np.nan, np.nan, np.nan