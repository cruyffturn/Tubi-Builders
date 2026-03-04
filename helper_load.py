import pickle
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
#import networkx as nx

from joblib import Parallel, delayed, parallel_backend

import helper_em_tf
#import helper
#import helper_dag

from helper_prob.models.helper_mvn import get_KL
from helper_prob.models.helper_mvn import get_ratio_ideal

from etc import other_favs

#used by get_p2
import helper_partial_data 

label0 = 'lb'
label1 = r'KL($\theta_p||\hat{\theta}$)'
label2 = r'KL($\theta_{\alpha}||\hat{\theta}$)'
label3 = r'HD($\hat{\mathcal{G}},\mathcal{G}_{\alpha}$)'
label4 = r'HD($\hat{\mathcal{G}},\mathcal{G}_{p}$)'
label5 = 'Adv. Success'
label6= r'HD($\hat{\mathcal{G}},\mathcal{G}_{ref}$)'
label7= r'undir-HD($\hat{\mathcal{G}},\mathcal{G}_{ref}$)'
hue = r'Init./$\epsilon$'

init_dic = {0:'True',
            1:'Ident.',
            3:'Random(*)',
            4:'Emp. Diag.',
            5:'Random',
            6:'IW(*)'
            }

#N_jobs = helper_local.n_jobs
#N_jobs = int(os.environ['MAX_CORES'])
#%%
def get_baseline(X,
                 pc_alpha = 0.01,
                 lambda1 = 0.1):
    
    load = pc(X, pc_alpha, 
              fisherz, True, 0,
              -1)    
    
    A_pc = helper.get_adj(load)
                
    loss_type = 'l2'
    W_nt = notears_linear(X, lambda1, loss_type, 
                          max_iter=100, 
                          h_tol=1e-8, 
                          rho_max=1e+16, 
                          w_threshold=0.3)
    
    A_nt = (np.abs(W_nt)>0).astype(int)
                
    return A_pc, A_nt
                
def get_stats(A_p, W_p, 
              mu_p, S_p,
              mu_a, S_a,
              idx_s, idx_t):
    
    avg_deg = np.mean(np.sum(A_p,0))    #?
    
#    idx_s, idx_t = idx_adv_train[:2]
    idx_copa = np.where(A_p[:,idx_t])[0]
    idx_copa = idx_copa[idx_copa!=idx_s]
    
    n_copa = len(idx_copa)    
    weight = W_p[idx_s,idx_t]
    
    P_p = helper_dag.get_pear(S_p)
    partial = P_p[idx_s,idx_t]
    
    KL_min = get_KL(mu_p, S_p, mu_a, S_a)   #Computes KL(p || a)
    
    n_in = np.sum(A_p[:,idx_s])
    n_out = np.sum(A_p[idx_s])
#    n_out_s = np.sum(A_p[idx_t])
    
#    S_norm_p = getCorr_XY(S_p, S_p, 
#                          return_sigma = False)
    
    std = np.sqrt(np.diag(S_p))
    denom = np.outer(std ,std)
        
    S_norm_p = S_p / denom
        
    corr = S_norm_p[idx_s,idx_t]
#    import ipdb;ipdb.set_trace()
    
    return avg_deg, n_copa, weight, \
            partial, KL_min, n_out, n_in, \
            corr

def get_prob(df):
    
    bool_mat = np.zeros((len(df),int(np.log2(len(df)))),bool)
    prob = np.zeros(len(df))
    for i, row_i in df.iterrows():
        prob[i] = row_i.iloc[1]
#        for ii, bool_ in enumerate(row_i.iloc[0].split(',')):
        for ii, bool_ in enumerate(str(row_i.iloc[0]).split(',')):
#            import ipdb;ipdb.set_trace()
#            bool_mat[i,ii] = int(bool_)            
            bool_mat[i,ii] = int(float(bool_))
    
    prob_sum = np.zeros(bool_mat.shape[1])
    for i in range(bool_mat.shape[1]):
        prob_sum[i] = np.sum(prob[bool_mat[:,i]])
    
    return prob_sum
                
def get_savePath(savePath,
                 bool_mcar, 
                 exp_type,
                 bool_short = False):
    
    savePath2 = os.path.join(savePath,'fig')    
    if bool_mcar:
        savePath2 = os.path.join(savePath2,'mcar')        
        
    savePath2 = os.path.join(savePath2, 'exp_type_%i'%exp_type)
    
    if exp_type == 0 and bool_short:
        savePath2 = os.path.join(savePath2, 'short')
        
    return savePath2

def main(savePath,
         X, model, 
         S, mu,
         A_p, A_a,
         S_a, mu_a,
         idx_adv_train,
         seed_model,         
         #Configuration parameters
         bool_mcar, 
         exp_type,
         bool_init,
         n_rep = 20,
         bool_sub = True,
         debug = 0,
         bool_short = False,
         idx_mask = None,
         idx_s = None, 
         idx_t = None,
         legacy_random = True,
         **kwargs_mlr):    
    
    '''
    bool_short:         #missDAG runs only on a specific initialization
    legacy_random:      #Used for debugging purposes
    '''
    if idx_mask is None:
        idx_mask = idx_adv_train
    else:
        print('helper_load.main using idx_mask')
        
    if debug:
        n_rep = 1
#    savePath2 = os.path.join(savePath,'fig')    
#    if bool_mcar:
#        savePath2 = os.path.join(savePath2,'mcar')
#        
#    savePath2 = os.path.join(savePath2, 'exp_type_%i'%exp_type)
    
    savePath2 = get_savePath(savePath,
                             bool_mcar, 
                             exp_type,
                             bool_short = bool_short)
    
    if not os.path.exists( savePath2):
    	os.makedirs( savePath2)
        
    
    #Sets the seed
    np.random.seed(42)
    print('tf seed', seed_model)
    tf.random.set_seed(seed_model)
    
    p_r_x, p2 = get_p2(X, model, bool_sub,
                       idx_adv_train,
                       bool_mcar)
    
    #Saves the prob
    save_prob(savePath2,
              p_r_x, 
              idx_mask,
              bool_full = True)
    
#    import ipdb;ipdb.set_trace()
    if exp_type == 0:
        df_l, err_W_l, \
        A_est_all_l, hue_l = missdag(X, p2,         
                                   S, mu,
                                   A_p, A_a,
                                   S_a, mu_a,
                                   idx_mask,
                                   bool_init,
                                   n_rep,
                                   bool_short = bool_short,
                                   idx_s = idx_s, 
                                   idx_t = idx_t,
                                   legacy_random = legacy_random,
                                   **kwargs_mlr
                                   )
    else:
        df_l, A_est_all_l, \
        hue_l = impute(X, p2,         
                       S, mu,
                       A_p, A_a,
                       idx_mask,
                       n_rep,
                       idx_s = idx_s, 
                       idx_t = idx_t)
        
        err_W_l = None
        
        
    draw(savePath2,
         p_r_x, 
         idx_adv_train,
         savePath,
         df_l,
         err_W_l,
         bool_init,
         bool_mcar, 
         exp_type,
         A_est_all_l, 
         hue_l
         )

def get_p2(*args,
           bool_partial_read = False,
           **kwargs):
    
    if not bool_partial_read:
        return get_p2_all_read(*args,**kwargs)
    else:
        return get_p2_partial_read(*args,**kwargs)
    
        
def get_p2_all_read(X, model, bool_sub,
                   idx_adv_train,
                   bool_mcar,
                   bool_full = 1,
                   bool_omit_data = None,
                   training = True):
    
    '''
    In:
        idx_adv_train:  p_sub, #selects the model input
    '''
    N = len(X)    
#    p = X.shape[1]
    
    if not bool_sub:
        p_r_x = model(X, training=training)  # Forward pass
    else:    
        p_r_x = model(X[:,idx_adv_train], training=training)  # Forward pass

    if not bool_mcar:
        if not bool_full:
            p_r_x = p_r_x.reshape(-1)
            p2 = np.stack([1-p_r_x,p_r_x],1)
        else:
            p2 = p_r_x
    else:
        if bool_full:
            if bool_omit_data is None:
                p_r = p_r_x.numpy().mean(0)[np.newaxis,:]
                p2 = np.repeat(p_r,N,0)
            else:
                if p_r_x.shape[1] == 1:
                    raise ValueError
                else:
                    p_r_0 = p_r_x.numpy()[bool_omit_data.astype(bool)].mean(0)
                    p_r_1 = np.repeat(p_r_0[np.newaxis,:], N, 0)
                    
                    p2 = helper_partial_data.get_omit_portion(p_r_1, 
                                                              bool_omit_data)
    
    return p_r_x, p2        
        
def missdag(X, p2,         
            S, mu,
            A_p, A_a,
            S_a, mu_a,
            idx_mask,
            bool_init,
            n_rep,
            lambda1 = 0.1,   #NOTEARS lambda
            bool_full = 1,
            bool_short = False,
            return_err = True,
            idx_s = None,
            idx_t = None,
            legacy_random = True,
            bool_hat_first = False,
            **kwargs
            ):
        
    
    if bool_init:
        if not bool_short:
            init_L = [0,1,3,4,6]
        else:
            print('short search')
#            init_L = [3,4]
            init_L = [3]
            
    #    init_L = [1,3,4]
    #    init_L = [0,6]
        #eps_L = [.005, .001,.0005]
        eps_L = [.001,]
    else:
        init_L = [0,3]
        eps_L = [.005, .001,.0005]

    
    df_L = []
    
    bool_history = 0
    like_all_l_param = []
    err_W_l = []
    
    W_est_all_l = []
    
    bool_sparse = 0
    bool_dag = 1
#    bool_dag = 0

    temp_folder = None    
    with parallel_backend('multiprocessing',
#    	                      n_jobs = n_jobs,
                              n_jobs = 1
                              ):
                                    
#        trainable = delayed(helper.multiple_em)
        trainable = delayed(helper.multiple_em_par)
        load_L = Parallel(verbose=11,
        	                      temp_folder=temp_folder)(trainable(X, p2, n_rep,
                                                          init_mode,
                                                          bool_full, S, mu,
                                                           idx_mask = idx_mask,
                                                           bool_sparse = bool_sparse,
                                                           alpha = None,
                                                           bool_while = True,
                                                           eps = eps,
                                                           bool_history = bool_history,
                                                           bool_dag = bool_dag,
                                                           verbose = False,
                                                           n_jobs = N_jobs,
                                                           lambda1=lambda1,
                                                           legacy_random=legacy_random,
                                                           **kwargs)
        	                    for init_mode in init_L
                                    for eps in eps_L)
                        
    hue_l = [(init_mode,eps) for init_mode in init_L for eps in eps_L]
    for load,(init_mode,eps) in zip(load_L,hue_l):
            
        if not bool_history:
            
            if not bool_dag:
                mu_est_all, S_est_all, \
                K_est_all, lb_all = load
                
#                    n_total = helper.get_graph_error_total(K_a, K_est_all, 0.1)
                
            else:
                mu_est_all, S_est_all, \
                W_est_all, lb_all = load
                

                A_est_all = (np.abs(W_est_all)>0).astype(int)
                
                n_total_a = helper.get_pdag_dist(A_a, A_est_all)
                n_total_p = helper.get_pdag_dist(A_p, A_est_all)
                
                if not return_err:
                    err_rate = helper.get_adv_err(A_p, A_a, A_est_all)
                    succ_rate = 1-err_rate
                                    
                else:
                    err_rate, idx_edges = helper.get_adv_err(A_p, A_a, A_est_all, 
                                                             return_err = True)                                                
                    succ_rate = 1-err_rate
                    
                    bool_sel = (idx_edges[:,0] == idx_s) & \
                                (idx_edges[:,1] == idx_t)
                    
                    succ_rate = succ_rate[:,bool_sel][:,0]                        
        
#                    n_total_a = helper.get_graph_error_dag(W_a, W_est_all)
#                    n_total_p = helper.get_graph_error_dag(W_a, W_est_all)
                
            stats = helper.get_stats(mu_est_all, S_est_all,
                                     mu_a, S_a, 
                                     mu, S,
                                     bool_hat_first = bool_hat_first)
            
#            n_rem, n_add = helper.get_graph_error(K_a, K_est_all, 0.1)   
#            n_total = n_rem+n_add                            
            
            if bool_hat_first:
                raise ValueError
            df = pd.DataFrame(stats, columns = [label1, label2])
        #    temp = 
            if eps == .001:
                eps_str = 'same'
            elif eps < .001:
                eps_str = 'strict'
            elif eps > .001:
                eps_str = 'loose'
                
            df[hue] = init_dic[init_mode]+'/'+ eps_str# + '/'+f'{eps/100:.0E}'
            if bool_dag:
                df[label3] = n_total_a
                df[label4] = n_total_p
                df[label5] = succ_rate
            df[label0] = lb_all
            
            
            df_L.append(df)
            
#                err_W_l.append(np.mean(np.abs(W_est_all-W_a),0))
            err_W_l.append((A_est_all!=A_a).mean(0))
            W_est_all_l.append(W_est_all)
            
        else:
            like_all_l = load
            like_all_l_param.append(like_all_l)
            
    return df_L, err_W_l, W_est_all_l, hue_l
            

def impute(X, p2,         
           S, mu,
           A_p, A_a,
           idx_mask,
           n_rep,
           lambda1 = 0.1,   #NOTEARS lambda
           bool_full = 1,
#           return_err = False
           return_err = 1,
           idx_s = None,
           idx_t = None,
           ):
    
    #Exp type 1
    
    hue_est = 'impute/method'
    df_l = []
    
    A_est_all_l = []
    
    hue_l = []
    
    for bool_impute in [0,1]:
        
        if not bool_impute:
            mode_est_L = ['pc']
        else:
            mode_est_L = ['pc','nt']
        
        for mode_est in mode_est_L:
            A_est_all = helper.multiple_est(X, p2, n_rep,
                                            bool_full, 
                                            bool_impute,
                                            mode_est,
                                            idx_mask = idx_mask,
                                            n_jobs = N_jobs,
                                            lambda1=lambda1
                                            )
            
            if mode_est == 'nt':
                A_est_all = (np.abs(A_est_all)>0).astype(int)                        
            
            n_total_a = helper.get_pdag_dist(A_a, A_est_all, allow_pdag = True)
            n_total_p = helper.get_pdag_dist(A_p, A_est_all, allow_pdag = True)
            
            if not return_err:
                err_rate = helper.get_adv_err(A_p, A_a, A_est_all, 
                                              allow_pdag = True)
                succ_rate = 1-err_rate
                                
            else:
                err_rate, idx_edges = helper.get_adv_err(A_p, A_a, A_est_all, 
                                              allow_pdag = True,
                                              return_err = True)                                                
                succ_rate = 1-err_rate
                
                bool_sel = (idx_edges[:,0] == idx_s) & (idx_edges[:,1] == idx_t)
                
                succ_rate = succ_rate[:,bool_sel][:,0]
                
            df = pd.DataFrame(np.stack([n_total_a,n_total_p,succ_rate],1), 
                                  columns = [label3, label4,label5])
            
            df[hue_est] = str(bool(bool_impute))+'/'+mode_est
            
            df_l.append(df)
            
            A_est_all_l.append(A_est_all)
            hue_l.append((bool_impute,mode_est))
            
    return df_l, A_est_all_l, hue_l
            
#%% Draws the figures and exports the tables
def save_prob(savePath2,
              p_r_x, 
              idx_mask,
              bool_full = True,
              bool_tf = True):
    
    #Saves the probabilities
    if bool_full:
        if bool_tf:
            p_r = p_r_x.numpy().mean(0)
        else:
            p_r = p_r_x.mean(0)
            
        obs_mat = helper_em_tf._get_obs_mat(len(idx_mask)).astype(int)
        row_L = [','.join(row.astype(str).tolist()) for row in obs_mat]
        
    df_prob = pd.DataFrame({'observed indices (%s)'%idx_mask:row_L,
                            'probability':p_r})
    df_prob.to_csv(os.path.join(savePath2,'p_r.csv'),
                           index=False)
def draw(savePath2,
         p_r_x, 
         idx_adv_train,
         savePath,
         df_l,
         err_W_l,
         bool_init,
         bool_mcar, 
         exp_type,
         A_est_all_l, 
         hue_l,
         bool_full = True):
    
    
    #Saves the figures
    if exp_type == 0:
        draw_missdag(df_l,
                     err_W_l,
                     savePath2,
                     bool_init)
    else:
        draw_impute(df_l,
                    savePath2,
                    bool_init)
    
    with open(os.path.join(savePath2,'A_est.p'), "wb") as f:
        pickle.dump([A_est_all_l, hue_l], f)
            
def draw_missdag(df_l,
                 err_W_l,
                 savePath2,
                 bool_init,
                 bool_dag = True,
                 bool_history = False,
                 bool_sparse = False):
    
    with open(os.path.join(savePath2,'results_exp_type_0.p'), "wb") as f:
    	pickle.dump(df_l, f)
        
    figname = ''
    
    if bool_sparse:
        figname = 'sparse_' + figname
        
    if bool_init:
        figname = 'init_'
    else:
        figname = 'eps_'
        
    if not bool_history:	
        df_all = pd.concat(df_l, ignore_index = 1)

        df_mean = df_all.groupby(hue).mean()
        df_std = df_all.groupby(hue).std()
        
        summary = pd.concat([df_mean,df_std], axis = 1)
        summary.to_csv(os.path.join(savePath2,'summary.csv'))
		
        n_ax = 4
        fig, ax = plt.subplots(1,n_ax)
        
        sns.scatterplot(data=df_all, x = label1, y = label2, hue = hue,
                        ax = ax[0])
        
        if bool_dag:
            sns.countplot(data=df_all, x = hue, hue = label3,
                          ax = ax[1])
            
            sns.countplot(data=df_all, x = hue, hue = label4,
                          ax = ax[2])
            
            sns.countplot(data=df_all, x = hue, hue = label5,
                          ax = ax[3])
        else:
            figname = figname+'_debug'                
        
        
        fig.set_size_inches( w = n_ax*10,h = 5)	
        fig.savefig(os.path.join(savePath2,figname+'denea.png'), 
        	            dpi=200, bbox_inches='tight')        
        
        fig, ax = plt.subplots(1,len(err_W_l))
        if len(err_W_l) == 1:
            ax = [ax]        
        for ax_i, err_W in zip(ax,err_W_l):
            sns.heatmap(err_W, 
#                        xticklabels=columns, 
#                        yticklabels=columns,
                        ax = ax_i,
                        square = True,
                        center= 0.5,
                        vmin = 0,
                        vmax = 1.,
                        annot = True,
                        fmt=".2f",
                        )
        

        fig.set_size_inches( w = len(ax)*8,h = 5)
        fig.savefig(os.path.join(savePath2,figname+'W_err.png'), 
        	            dpi=200, bbox_inches='tight')
    
    else:
        fig, ax = plt.subplots()
        i = 0
        import matplotlib.cm as cm
        total = len(like_all_l_param)*n_rep
        for like_all_l in like_all_l_param:
            for like_ in like_all_l:
                c = cm.Blues(i/total,1)
                ax.plot(like_, color=c)
                i+=1
                
        fig.savefig(os.path.join(savePath2,figname+'like_denea.png'), 
        	            dpi=200, bbox_inches='tight')
        
def draw_impute(df_l,
                savePath2,
                bool_init,
                bool_dag = True,
                bool_history = False,
                bool_sparse = False):
        
    hue_est = 'impute/method'
    
    with open(os.path.join(savePath2,'results_exp_type_1.p'), "wb") as f:
    	pickle.dump(df_l, f)
        
    df_all = pd.concat(df_l, ignore_index = 1)
    
    df_mean = df_all.groupby(hue_est).mean()
    df_std = df_all.groupby(hue_est).std()
        
    summary = pd.concat([df_mean,df_std], axis = 1)
    summary.to_csv(os.path.join(savePath2,'summary_exp_type_1.csv'))


    fig, ax = plt.subplots(1,3)
    
    sns.countplot(data=df_all, x = hue_est, hue = label4,
                  ax = ax[1])
    sns.countplot(data=df_all, x = hue_est, hue = label3,
                  ax = ax[0])    
    sns.countplot(data=df_all, x = hue_est, hue = label5,
                  ax = ax[2])
    
    fig.set_size_inches( w = 15,h = 5)
    fig.savefig(os.path.join(savePath2,'2_''denea.png'),
                dpi=200, bbox_inches='tight')
    
    temp = df_all.groupby(hue_est)[label4].value_counts(normalize=1)
    temp.plot(kind='bar',stacked=True)

def _draw(A, ax):
    
    G = nx.DiGraph(A)
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")
    
#    fig, ax = plt.subplots(2)
    nx.draw_networkx(G, pos=pos, ax=ax)


def draw_g(A_p,A_a,title):
        
    fig, ax = plt.subplots(1,2)
    _draw(A_p, ax[0])
    _draw(A_a, ax[1])
    
    ax[0].set_title(title)
    
    return fig

def draw_g2(A, W, idx_s,idx_t,title):

    fig, ax = plt.subplots()
    
    G = nx.DiGraph(A)
    
    colors_l = []
    weight_l = []
    
    edges = G.edges()
    for u,v in edges:
#        import ipdb;ipdb.set_trace()
        if u == idx_s and v == idx_t:
            color = 'r'
        else:
            color = 'g'       
            
        colors_l.append(color)
        weight_l.append(np.abs(W[u,v]))
        


#    G = nx.DiGraph(A)
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")
    
#    fig, ax = plt.subplots(2)
#    nx.draw(G, pos=pos, 
##            edges=edges, 
#            edge_color=colors_l, 
#            width=weight_l,
#            ax=ax)
    
    ax.set_title(title)
    nx.draw_networkx(G, pos=pos,
#                     edges=edges, 
                     edge_color=colors_l, 
                     width=weight_l,
                     ax=ax)
    
    return fig
    

def save_rel(A_pc, A_nt, 
             A_p, savePath,
             exp_type, bool_mcar,
             bool_short = False
             ):
    
#    import ipdb;ipdb.set_trace()
    savePath2 = get_savePath(savePath,
                             bool_mcar, 
                             exp_type,
                             bool_short = bool_short)
    
    with open(os.path.join(savePath2,'A_est.p'), "rb") as f:
        A_est_all_l, hue_l = pickle.load(f)
    
    df_l = []
    
    if exp_type == 0:
        A_ref = A_nt
        allow_pdag = False
        hue_sub = hue
    else:
        hue_sub = 'impute/method'
        allow_pdag = True
        
    for A_est_all, hue_i in zip(A_est_all_l,hue_l):
        
        if exp_type == 0:
            init_mode, eps = hue_i            
            A_est_all = (np.abs(A_est_all)>0).astype(int)            
            
        else:
            bool_impute, mode_est = hue_i            
            if mode_est == 'nt':
                A_est_all = (np.abs(A_est_all)>0).astype(int)
                A_ref = A_nt
                
            else:
                A_ref = A_pc                        
            
        n_total_rel = helper.get_pdag_dist(A_ref, A_est_all, 
                                           allow_pdag=allow_pdag)
        n_total_p = helper.get_pdag_dist(A_p, A_est_all,
                                         allow_pdag=allow_pdag) #for reference.        
                        
        df = pd.DataFrame({label6:n_total_rel,
                           label4:n_total_p})
        
        if exp_type == 0:
#            import ipdb;ipdb.set_trace()
            A_est_all_und = (A_est_all.astype(int) + \
                             np.transpose(A_est_all,[0,2,1]).astype(int))
            A_est_all_und = (np.abs(A_est_all_und)>0)
            
            n_total_rel_und = helper.get_pdag_dist(A_ref, A_est_all_und, 
                                                   allow_pdag=True)
            
            df[label7] = n_total_rel_und
            
            
        if exp_type == 0:
            if eps == .001:
                eps_str = 'same'
            elif eps < .001:
                eps_str = 'strict'
            elif eps > .001:
                eps_str = 'loose'
                
            df[hue_sub] = init_dic[init_mode]+'/'+ eps_str# + '/'+f'{eps/100:.0E}'
        else:
            df[hue_sub] = str(bool(bool_impute))+'/'+mode_est
            
        df_l.append(df)
                    
    df_all = pd.concat(df_l, ignore_index = 1)

    df_mean = df_all.groupby(hue_sub).mean()
    df_std = df_all.groupby(hue_sub).std()
    
    summary = pd.concat([df_mean,df_std], axis = 1)
    summary.to_csv(os.path.join(savePath2,'rel_summary.csv'))
    
    
def get_row_names(idx_adv, names_adv):
    
    '''
    idx_adv:    p_sub,
    names_adv:  p_sub, #[i] is name of idx_adv[i]
    '''
    
    p_sub = len(idx_adv)
    row_l = []
    
    for i, idx_temp in enumerate(helper_em_tf.powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), 
                                 idx_sub_m).astype(np.int32)
        
        if len(idx_sub_o)>0:
            str_ = ' & '.join(names_adv[idx_sub_o].tolist())
        else:
            str_ = 'None'
            
        row_l.append(str_)
                    
    return row_l

def get_ratios(mu, S, 
               mu_a, S_a, 
               X,
               idx_adv_train):
    
    ratio_l = []
    for i in range(3):
        
        if i == 0:
            idx_sub_0 = [0]
        elif i == 1:
            idx_sub_0 = [1]
        elif i == 2:
            idx_sub_0 = [0,1]
            
        X_sub = X[:,idx_sub_0]
        idx_sub = idx_adv_train[idx_sub_0]
        
        ratio = get_ratio_ideal(mu_a[idx_sub], 
                                S_a[idx_sub][:,idx_sub], 
                                mu[idx_sub], 
                                S[idx_sub][:,idx_sub], 
                                X_sub)
        ratio_l.append(ratio)
    
    return ratio_l

def draw_ratio(drawPath2,
               model, 
               X, mu, S,                 
               idx_adv_train,
               idx_mask,
               mu_a, S_a,
               idx_s, idx_t,
               str_1
               ):    

    X_sub = X[:,idx_adv_train]
    temp = np.stack([X_sub.min(0),
                     X_sub.max(0)])
        
    x1_pts = np.arange(-8,7.9,0.1)
    x2_pts = x1_pts
#                    x1_pts = np.arange(temp[0,0],temp[0,1],0.1)
#                    x2_pts = np.arange(temp[1,0],temp[1,1],0.1)
    
    X_grid = other_favs.get_mesh_ravel([x1_pts, x2_pts])
    ratio = get_ratio_ideal(mu_a[idx_adv_train], 
                            S_a[idx_adv_train][:,idx_adv_train], 
                            mu[idx_adv_train], 
                            S[idx_adv_train][:,idx_adv_train], 
                            X_grid)
                        
    p_r_x, _ = get_p2(X_grid, model, 
                      bool_sub=False,
                      idx_adv_train=None,
                      bool_mcar=False,
                      bool_full=1)
    p_r_x = p_r_x.numpy()
    
    
    names_mask = np.array([str(idx_i) for idx_i in idx_mask])
    names_mask[idx_mask==idx_s] = 'S'
    names_mask[idx_mask==idx_t] = 'T'
    
    names_row = get_row_names(idx_mask, 
                                          names_mask)
    
#    if not np.array_equal(idx_adv_train,idx_mask):
#        raise TypeError
        
#    if 0:
#        fig = helper_draw.draw_2d_long(X_grid, p_r_x, 
#                                       ratio,
#                                       names_row,
#                                       bool_full = 1)
    if p_r_x.shape[1] == 4:
        ratio_l = get_ratios(mu, S, 
                             mu_a, S_a, 
                             X_grid,
                             idx_adv_train)
        
        names_ratio = ['x_s','x_t','x_s,x_t']
        ratio_l = [ratio_l[temp] for temp in [2,1,0]] 
        names_ratio = [names_ratio[temp] for temp in [2,1,0]]
        fig = helper_draw.draw_2d_long_mul(X_grid, p_r_x, 
                                           ratio_l,
                                           names_row,
                                           names_ratio,
                                           bool_full = 1)
    elif p_r_x.shape[1] == 2:
        
        names_ratio = ['x_s,x_t']
        
        fig = helper_draw.draw_2d_v2(X_grid, p_r_x, 
                                       ratio,
                                       names_row,
                                       names_ratio,
                                       bool_full = 1)
        
#        pass
#                    fig = helper_draw.draw_2d(X_grid, p_r_x, 
#                                              ratio, 
#                                              bool_full=1, 
#                                              idx_mask=0)
    
    fig.suptitle(str_1)

    if p_r_x.shape[1] == 4:
        fig.set_size_inches( w = 20,h = 10)
    elif p_r_x.shape[1] == 2:    
        fig.set_size_inches( w = 10,h = 5)
        
    drawfile = os.path.join(drawPath2,'ratio.png')
    fig.savefig(drawfile, dpi=200, bbox_inches='tight')
    
    return drawfile

def draw_g2_edge(A_est, W, 
                 idx_s, idx_t,
                 ax,
                 node_names = None,
                 bool_mcar = None,
                 hue = None,
                 type_graph = None,
                 exp_type = None):
    '''
    Uses https://networkx.org/documentation/stable/auto_examples/graph/plot_dag_layout.html
    '''
#    fig, ax = plt.subplots()
    
    G = nx.DiGraph(np.round(A_est,2))    
        
    colors_l = []
    weight_l = []
    
    edges = G.edges()
    for u,v in edges:
#        import ipdb;ipdb.set_trace()
        if u == idx_s and v == idx_t:
            color = 'r'
        elif W[u,v] != 0:
            color = 'g'
        else:
            color = 'k'
            
        colors_l.append(color)
        weight_l.append(np.abs(W[u,v]))
        

    if node_names is not None:
        G = nx.relabel_nodes(G, dict(zip(range(len(W)),node_names)))
        
#    G = nx.DiGraph(A)
    G_true = nx.DiGraph(W)
    
    if type_graph != 'sachs':
    #    for layer, nodes in enumerate(nx.topological_generations(G)):
        for layer, nodes in enumerate(nx.topological_generations(G_true)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                G.nodes[node]["layer"] = layer

    if node_names is None:
        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(G, subset_key="layer",
#                                     scale = 1
                                     )
    else:
        pos = get_sachs_pos()
    
    if type_graph == 'chain':
        pos[idx_s][1] += 0.25
#    fig, ax = plt.subplots(2)
#    nx.draw(G, pos=pos, 
##            edges=edges, 
#            edge_color=colors_l, 
#            width=weight_l,
#            ax=ax)
    
#    ax.set_title(title)
#    import ipdb;ipdb.set_trace()
    
    nx.draw_networkx_nodes(G, 
                           pos=pos,
    #                     width=weight_l,
                         ax=ax,
#                         with_labels=True
                         )
    nx.draw_networkx_labels(G,
                            pos=pos,
                            ax=ax)
    nx.draw_networkx_edges(G, 
                           pos=pos,
                           alpha=0.5,
                         edge_color=colors_l,
    #                     width=weight_l,
                         ax=ax)
    
    
#    nx.draw_networkx(G, pos=pos,
##                     edges=edges, 
#                     edge_color=colors_l, 
#                     
##                     width=weight_l,
#                     ax=ax)
    
#    if node_names is None:
    kws = dict(verticalalignment='bottom',
               bbox=dict(facecolor='white', alpha=0., linewidth=0)
#                   alpha=0.2
               )
#    else:
#        kws = {}
    nx.draw_networkx_edge_labels(G, pos,ax=ax,
                                 edge_labels=nx.get_edge_attributes(G,'weight'),
                                 font_size = 10,
                                 **kws,
#                                 connectionstyle="arc3,rad=0.1"
                                 )
    
    if bool_mcar is not None:
        title = ['MNAR','MCAR'][bool_mcar]
    else:
        title = ''
        
    if hue is not None:
        if exp_type == 0:
            
            init_mode, eps = hue
            if eps == .001:
                eps_str = 'same'
            elif eps < .001:
                eps_str = 'strict'
            elif eps > .001:
                eps_str = 'loose'
            
            title = title + ' MissDAG:' + init_dic[init_mode]+'/'+ eps_str
        else:    
            title = title + ' MissPC:' + 'impute/method: '+str(hue)

    ax.set_title(title)
    

#    return fig

def get_sachs_pos():
#G = nx.relabel_nodes(G, dict(zip(range(len(W)),columns)))
    pos = {'raf':(-1,-0.5),
            'mek':(-1,-2),
            'plc':(1.5,1),
            'pip2':(2,-1.5),
            'pip3':(2,0),
            'erk':(-1,-3),
            'akt':(-1,-4),
            'pka':(0,-0.5),
            'pkc':(0,1.5),
            'p38':(1,-1),
            'jnk':(0.5,-1)
            }
    
    return pos

def get_mask(X, model, 
              seed_model,
              idx_adv_train,
              idx_mask,
              n_rep = 20,                    
              bool_mcar = False,
              bool_omit_data = None,
              training = True
              ):
    
    bool_full = True
#    bool_mcar = False
    bool_sub = True
    
    np.random.seed(42)
    print('tf seed', seed_model)
    tf.random.set_seed(seed_model)        
    
    p_r_x, p2 = get_p2(X, model, bool_sub,
                       idx_adv_train,
                       bool_mcar,
                       bool_omit_data = bool_omit_data,
                       training = training)
    
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cat = tf.random.categorical(np.log(p2), n_rep).numpy()#[:,0]

    mask = np.zeros((n_rep,)+X.shape,bool)
    
    for i in range(n_rep):
        
        cat_i = cat[:,i]
        
        if not bool_full:
    #            mask = cat.reshape(x.shape).astype(bool)#[:,0]
    #            mask = ~mask
            mask[i] = helper_em_tf.get_mask_nfull(cat_i, X.shape[1], 
                                                  idx_mask)
        else:
            mask[i] = helper_em_tf.get_mask_full(cat_i, X.shape[1],
                                                 idx_mask)
    
    print('% missing ' +'%.2f'%(np.mean(mask)*100))
    print('% missing target' +'%.2f'%(np.mean(mask[:,:,idx_mask])*100))
    print('% missing per column', mask[:,:,idx_mask].mean((0,1))*100)    
    
    return mask

def get_mask_rs(p_r_x,
                p,
                idx_mask,
                n_rep = 20,                    
                bool_mcar = False):
    
    N = p_r_x.shape[0]
    bool_full = 1
    seed = 42
    
    p2 = get_p2_rs(p_r_x,
                   bool_mcar)
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cat = tf.random.categorical(np.log(p2), n_rep).numpy()#[:,0]

    mask = np.zeros((n_rep,)+(N,p),bool)
    
    for i in range(n_rep):
        
        cat_i = cat[:,i]
        
        if not bool_full:
    #            mask = cat.reshape(x.shape).astype(bool)#[:,0]
    #            mask = ~mask
            mask[i] = helper_em_tf.get_mask_nfull(cat_i, p, 
                                                  idx_mask)
        else:
            mask[i] = helper_em_tf.get_mask_full(cat_i, p,
                                                 idx_mask)
    
    print('% missing ' +'%.2f'%(np.mean(mask)*100))
    print('% missing target' +'%.2f'%(np.mean(mask[:,:,idx_mask])*100))
    print('% missing per column', mask[:,:,idx_mask].mean((0,1))*100)    

    return mask

def get_p2_rs(p_r_x,
            bool_mcar,
            bool_full = 1):
    
    '''
    In:
        
    '''
    N = p_r_x.shape[0]

    if not bool_mcar:
        if not bool_full:
            raise ValueError
        else:
            p2 = p_r_x
    else:
        if bool_full:
            p_r = p_r_x.mean(0)[np.newaxis,:]
            p2 = np.repeat(p_r,N,0)
    
    return p2

def get_p2_partial_read(X_0, model, bool_sub,
                   idx_adv_train,
                   bool_mcar,
                   bool_full = 1,
                   bool_omit_data = None,
                   training = True):
    
    '''
    In:
        idx_adv_train:  p_sub, #selects the model input
    '''
    if not bool_full:
        raise ValueError
#    import ipdb;ipdb.set_trace()
    
    if bool_omit_data is None:
        raise ValueError
    else:
        bool_omit_data = bool_omit_data.astype(bool)
        
    X = X_0[bool_omit_data]

    N_sub = len(X)
    p_r_x_mnar = model(X, training=training).numpy()
    
    if not bool_mcar:
        p_r_x_0 = p_r_x_mnar
    else:
        p_r = p_r_x_mnar.mean(0)
        p_r_x_0 = np.repeat(p_r[np.newaxis,:], N_sub, 0)
                    
    p_r_x = np.zeros((len(bool_omit_data),p_r_x_0.shape[1]))
    p_r_x[:,0] = 1
    p_r_x[bool_omit_data] = p_r_x_0
    
    p_r_x = tf.constant(p_r_x)
    p2 = p_r_x
    
    return p_r_x, p2