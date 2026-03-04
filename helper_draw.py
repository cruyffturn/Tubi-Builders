# -*- coding: utf-8 -*-
'''
todo https://stackoverflow.com/questions/38973868/adjusting-gridlines-and-ticks-in-matplotlib-imshow
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
try:
    from helper_plot.plots import draw_heat
except:
    print('heatmap not available')


def draw_loop(corr_L, S_est_sel,
              mu_est_sel,
              loss_sel,
              prob_o_sel,
              prob_m_row_sel):
    
    '''
    Single model
    shape = (n_corr, n_model_no)
    
    S_est_sel:  3,shape
    mu_est_sel: 2, shape
    loss_sel:   shape,
    prob_o_sel: shape,
    prob_m_row_sel: shape,

    '''
        
    fig, ax = plt.subplots(1,3)
    i = 0
    model_dic = {0:'Lin',
                 1:'10 H',
                 2:'100 H'}
    
#    for i in range(norm_err.shape[1]):
    for ii, label in enumerate([r'$\hat{\Sigma}_{1,1}$',
                                r'$\hat{\Sigma}_{1,2}$',
                                r'$\hat{\Sigma}_{2,2}$']):
    
        ax[0].plot(corr_L, np.squeeze(S_est_sel[ii]), '*-', label = label)
    
    for ii, label in enumerate([r'$\hat{\mu}_{1}$',
                                r'$\hat{\mu}_{2}$']):
    
        ax[1].plot(corr_L, np.squeeze(mu_est_sel[ii]), '*-', label = label)
        
    ax[2].plot(corr_L, (1-prob_o_sel[:,i])*100, '*-', label = '% of missing entries')
    ax[2].plot(corr_L, (prob_m_row_sel[:,i])*100, '*-', label = '% of missing rows')
    
    ax[0].axline((0, 0), color = 'b', slope=1)
    ax[0].set_xlabel(r'$\Sigma_{\alpha,1,2}$')
    ax[0].set_ylabel(r'$\hat{\Sigma}$')
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[0].legend()
        
    ax[1].set_xlabel(r'$\Sigma_{\alpha,1,2}$')
    ax[1].set_ylabel(r'$\hat{\mu}$')
    ax[1].set_ylim([0,4])
    ax[1].legend()
    
    ax[2].set_xlabel(r'$\Sigma_{\alpha,1,2}$')
#    ax[2].set_ylabel('missiness')
    ax[2].legend()
    
    fig.set_size_inches( w = 17,h = 5)
    return fig
    #dict_sub = dict(corr=corr,
    #                model_cfg=model_cfg)
    #figname = '_'.join(['%s_%s'%(a,b) for a, b in dict_sub.items()])
#    figname = ''
#    fig.savefig(os.path.join(currPath,'loop_'+figname+'.png'), 
#    	            dpi=200, bbox_inches='tight')
    
def draw_loss(history, loss_type, bool_bivar = True):
    
    if bool_bivar:
        n_fig = 3
    else:
        n_fig = 2
        
    fig, ax = plt.subplots(1,n_fig)
    

    #ax.axhline(S[0,1],
    #           xmax=0.5,
    #           label = 'adv. target value'+sub_str,
    #           alpha= 0.2,
    ##           color = clrs[ii]
    #           )
#    if 
    if loss_type == 9 and 'plain_loss' in history.history.keys():
        ax[-2].plot(history.history['plain_loss'])                
    else:
        ax[-2].plot(history.history['loss'])
    
    ax[-2].set_xlabel('Training Epoch')
    
    if loss_type == 9:
        if 'plain_loss' in history.history.keys():
            loss_label = 'Avg. Loss'
        else:
            loss_label = 'Single Loss'
        
    ax[-2].set_ylabel(loss_label)
    
    ax[-1].plot((1-np.array(history.history['exp_obs_ratio']))*100, 
              label='% Missing')
    ax[-1].plot(np.array(history.history['p_miss_row'])*100,
               label='% Missing rows')
    #ax[-1].set_ylabel('Fraction')
    ax[-1].set_xlabel('Training Epoch')
    ax[-1].legend()
    
    if bool_bivar:
        if 'corr' in history.history.keys():
            ax[0].plot(history.history['corr'])
            
        ax[0].set_xlabel('Training Epoch')
        ax[0].set_ylabel(r'$\hat{\Sigma}_{1,2}$')
        
    fig.set_size_inches( w = int(n_fig*5),h = 5)
    return fig    

def draw_2d_long(X, p_r_x, ratio,
                 names_row,
                 bool_full = False):        
        
    if not bool_full:
        raise TypeError
    
    x1_pts = np.unique(X[:,0])
    x2_pts = np.unique(X[:,1])

    fig, ax = plt.subplots(2,4)
    
    
    ratio_grid = ratio.reshape(len(x1_pts),len(x2_pts))    
    
#    ratio_grid_stack = np.stack([ratio_grid,
#                                 ratio_est_grid])
    
#    log_ratio_grid_stack = np.log10(ratio_grid_stack)
    
    
    for i in range(2):
        for ii in range(4):
        
            if i == 0:                        
                draw = np.log10(ratio_grid)
                center = 0
                title = r'$\frac{P_{X,\theta_{\alpha}}(x)}{P_{X,\theta_p}(x)}$'
                vmin = np.min(draw)
                vmax = np.max(draw)
                
    #                        vmin = draw.min()
    #                        vmax = draw.max()
                
            if i == 1:
    #            draw = np.log10(ratio_est_grid)
                ratio_est_grid = p_r_x[:,ii]
                ratio_est_grid = ratio_est_grid.reshape(len(x1_pts),
                                                        len(x2_pts))
                draw = ratio_est_grid
                
                center = 0.5
                vmin = 0
                vmax = 1
                if not bool_full:
                    title = r'$P_{R\mid X,\phi}(1;x,\phi)$'
                else:
                    title = r'$P_{R\mid X,\phi}$'+'(%s observed)'%names_row[ii]
            
            
    
            draw_heat(draw, x1_pts, x2_pts, 
                      center=center, 
                      vmin=vmin, vmax=vmax, 
                      title =title, 
            #          name_0, name_1,
            #          name_modif, 
                      annot = False,
                      fmt = '%.2f',
                      n_skip_label = 10,
                      fig = fig,
                      ax = ax[i,ii],
                      bool_size = 0,
                      name_0=r'$X_s$'+' (Source)',
                      name_1=r'$X_t$'+' (Target)',
    #                  linewidths = 0.01,
    #                  linecolor = 'black',
                      square = True)
            
            if i == 0:
                break
        
    return fig

def draw_2d_long_mul(X, p_r_x, 
                     ratio_l,
                     names_row,
                     names_ratio,
                     bool_full = False):
        
    if not bool_full:
        raise TypeError
    
    x1_pts = np.unique(X[:,0])
    x2_pts = np.unique(X[:,1])

    fig, ax = plt.subplots(2,4)
    
        
    
#    ratio_grid_stack = np.stack([ratio_grid,
#                                 ratio_est_grid])
    
#    log_ratio_grid_stack = np.log10(ratio_grid_stack)
    
    
    for i in range(2):
        for ii in range(4):
        
            if i == 0:
                ratio = ratio_l[ii]
                ratio_grid = ratio.reshape(len(x1_pts),len(x2_pts))    
                draw = np.log10(ratio_grid)
                center = 0
#                title = r'$\log\frac{P_{X,\theta_{\alpha}}(x)}{P_{X,\theta_p}(x)}$'
                title = r'Density Ratio: $\log_{10} \Lambda(%s)$'%names_ratio[ii]#{P_{X,\theta_p}(x)}$
                vmin = np.min(draw)
                vmax = np.max(draw)
                
                if vmin == vmax:
                    vmax = vmin+1e-1
                
                ax_i = ax[i,ii]
                
    #                        vmin = draw.min()
    #                        vmax = draw.max()
                
            if i == 1:
    #            draw = np.log10(ratio_est_grid)
                ratio_est_grid = p_r_x[:,ii]
                ratio_est_grid = ratio_est_grid.reshape(len(x1_pts),
                                                        len(x2_pts))
                draw = ratio_est_grid
                
                center = 0.5
                vmin = 0
                vmax = 1
                if not bool_full:
                    title = r'$P_{R\mid X,\phi}(1;x,\phi)$'
                else:
                    title = r'$P_{R\mid X,\phi}$'+'(%s observed)'%names_row[ii]
                
                ax_i = ax[i,ii]
            
    
            draw_heat(draw, x1_pts, x2_pts, 
                      center=center, 
                      vmin=vmin, vmax=vmax, 
                      title =title, 
            #          name_0, name_1,
            #          name_modif, 
                      annot = False,
                      fmt = '%.2f',
                      n_skip_label = 10,
                      fig = fig,
                      ax = ax_i,
                      bool_size = 0,
                      name_0=r'$x_s$'+' (Source)',
                      name_1=r'$x_t$'+' (Target)',
    #                  linewidths = 0.01,
    #                  linecolor = 'black',
                      square = True)
            
            if i == 0 and ii==2:
                break
    
    ax[0, -1].axis('off')
    return fig

def draw_2d_v2(X, p_r_x, 
             ratio,
             names_row,
             names_ratio,
             bool_full = False):
        
    if not bool_full:
        raise TypeError
    
    x1_pts = np.unique(X[:,0])
    x2_pts = np.unique(X[:,1])

    if p_r_x.shape[1] == 2:
        fig, ax = plt.subplots(1,2)            
    
    for i in range(2):
        for ii in range(1):
        
            if i == 0:
#                ratio = ratio
                ratio_grid = ratio.reshape(len(x1_pts),len(x2_pts))
                draw = np.log10(ratio_grid)
                center = 0
#                title = r'$\log\frac{P_{X,\theta_{\alpha}}(x)}{P_{X,\theta_p}(x)}$'
                title = r'Density Ratio: $\log_{10} \Lambda(%s)$'%names_ratio[ii]#{P_{X,\theta_p}(x)}$
                vmin = np.min(draw)
                vmax = np.max(draw)
                
                if vmin == vmax:
                    vmax = vmin+1e-1
                
                ax_i = ax[i]
                
            if i == 1:
    #            draw = np.log10(ratio_est_grid)
#                ratio_est_grid = p_r_x[:,ii]
                ratio_est_grid = p_r_x[:,0]
                ratio_est_grid = ratio_est_grid.reshape(len(x1_pts),
                                                        len(x2_pts))
                draw = ratio_est_grid
                
                center = 0.5
                vmin = 0
                vmax = 1
                if not bool_full:
                    title = r'$P_{R\mid X,\phi}(1;x,\phi)$'
                else:
                    title = r'$P_{R\mid X;\phi}$'+'(%s observed)'%names_row[ii]
                
                ax_i = ax[i]
            
    
            draw_heat(draw, x1_pts, x2_pts, 
                      center=center, 
                      vmin=vmin, vmax=vmax, 
                      title =title, 
            #          name_0, name_1,
            #          name_modif, 
                      annot = False,
                      fmt = '%.2f',
                      n_skip_label = 10,
                      fig = fig,
                      ax = ax_i,
                      bool_size = 0,
                      name_0=r'$x_s$'+' (Source)',
                      name_1=r'$x_t$'+' (Target)',
    #                  linewidths = 0.01,
    #                  linecolor = 'black',
                      square = True)
            
            if i == 0 and ii==2:
                break
    
#    ax[0, -1].axis('off')
    return fig


def draw_beta(ax,history,idx_target,idx_omit):
    
    beta_history = np.stack(history.history['beta'])
    #beta_history = np.concatenate([beta[np.newaxis,:],beta_history],0)    
    
    ax.plot(beta_history[:,idx_target],
            label=r'$\tilde{\beta}_{t}$')
    ax.plot(beta_history[:,idx_omit])
    
    #ax.plot(beta[idx_omit],'*',color='tab:blue')
    #ax.plot(beta[idx_target],'*',color='tab:orange', 
    #        )
    ax.legend()
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('r$\tilde{\beta}$')

def draw_loss_beta(history):
            
    fig, ax = plt.subplots(1,2)
    

    #ax.axhline(S[0,1],
    #           xmax=0.5,
    #           label = 'adv. target value'+sub_str,
    #           alpha= 0.2,
    ##           color = clrs[ii]
    #           )
    ax[-2].plot(history.history['plain_loss']) 
    ax[-2].set_xlabel('Training Epoch')
    
    loss_label = 'Loss'
        
    ax[-2].set_ylabel(loss_label)
    
    ax[-1].plot((1-np.array(history.history['prob_obs']))*100, 
              label='% Local Missing')
    ax[-1].plot(np.array(history.history['p_miss_row'])*100,
               label='% Local Missing rows')
    ax[-1].set_ylabel('Missingness Rate')
    ax[-1].set_xlabel('Training Epoch')
    ax[-1].legend()    
        
    fig.set_size_inches( w = int(2*5),h = 5)
    return fig  

def draw_obj(history):
    
#    keys = ['loss','inner_loss',]
    if 'penalty_obj' not in history.history.keys():
        fig, ax = plt.subplots(1,2)

        ax[-2].plot(history.history['loss']) 
        ax[-2].set_xlabel('Training Epoch')
        
        loss_label = 'Upper level obj. function'
        
        ax[-2].set_ylabel(loss_label)
        
        ax[-1].plot(history.history['inner_loss'])
        ax[-1].set_ylabel('Lower level obj. function')
        ax[-1].set_xlabel('Training Epoch')
        ax[-1].legend()    

        fig.set_size_inches( w = int(2*5),h = 5)
    
    else:
        fig, ax = plt.subplots(1,4)
        
        ax[0].plot(history.history['penalty_obj'])
        ax[0].set_ylabel('Total objective with penalty')
        ax[0].set_xlabel('Training Epoch')
        
        ax[1].plot(history.history['loss']) 
        ax[1].set_xlabel('Training Epoch')        
        ax[1].set_ylabel('Upper level obj. function')
        
        ax[2].plot(history.history['inner_loss'])
        ax[2].set_ylabel('Lower level obj. function')
        ax[2].set_xlabel('Training Epoch')
        ax[2].legend()            
        
        ax[3].plot(history.history['q']) 
        ax[3].set_ylabel('Optimal Value Gap')
        ax[3].set_xlabel('Training Epoch')
        
        fig.set_size_inches( w = int(4*5),h = 5)

    return fig  

def draw_loss_clsf(history):
            
    fig, ax = plt.subplots(1,2)    

    #ax.axhline(S[0,1],
    #           xmax=0.5,
    #           label = 'adv. target value'+sub_str,
    #           alpha= 0.2,
    ##           color = clrs[ii]
    #           )
    ax[-2].plot(history.history['plain_loss']) 
    ax[-2].set_xlabel('Training Epoch')
    
    loss_label = 'Loss'
        
    ax[-2].set_ylabel(loss_label)
    
    ax[-1].plot((1-np.array(history.history['prob_obs']))*100, 
              label='% Local Missing')
    ax[-1].plot((1-np.array(history.history['p_r_y1']))*100,
               label='% Local Missing y=1')
    ax[-1].plot((1-np.array(history.history['p_r_y0']))*100,
               label='% Local Missing y=0')
    ax[-1].set_ylabel('Missingness Rate')
    ax[-1].set_xlabel('Training Epoch')
    ax[-1].legend()    
        
    fig.set_size_inches( w = int(2*5),h = 5)
    return fig  