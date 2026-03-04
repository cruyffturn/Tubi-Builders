# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, Normalize

#from sklearn.datasets import make_circles, make_classification, make_moons
#from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from helper_plot.plots import draw_heat

from helper_load import get_p2

def _get_grid(X,
              x_min, x_max,
              y_min, y_max):

    '''
    @chatgpt
    '''
    # Create a mesh grid with a resolution of 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # ----------------- Predict on the grid -----------------
    # Flatten the grid to pass to classifier for predictions then reshape back
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    shp = xx.shape
    
    return xx, yy, grid_points

def _get_grid_n(X,
                x_min, x_max,
                y_min, y_max,
                n = 100):

    '''
    @chatgpt
    '''
    # Create a mesh grid with a resolution of 0.02
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n),
                         np.linspace(y_min, y_max, n))
    
    # ----------------- Predict on the grid -----------------
    # Flatten the grid to pass to classifier for predictions then reshape back
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    shp = xx.shape
    
    return xx, yy, grid_points

def draw_decision_boundary(X, y, 
                          x_target, y_target,
                          model, ax,
                          x_min = None, x_max = None,
                          y_min = None, y_max = None,
                          bool_miss = None,
                          **kwargs):
    
    '''
    Draws the decision boundary of the NN
    '''
    
    if x_min is None:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
    xx, yy, grid_points = _get_grid(X,
                                    x_min, x_max,
                                    y_min, y_max)
    
    Z = model(grid_points).numpy()
#    Z = logits>0
    Z = Z.reshape(xx.shape)
    
#    import ipdb;ipdb.set_trace()

    # ----------------- Plotting -----------------
    #plt.figure(figsize=(10, 8))
    # Create a color map for the predictions
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
#    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
                                
    # Plot the decision boundary by assigning a color to each point in the mesh.
    if Z.min() < 0:
        _min = Z.min()
    else:
        _min = -1e-7
    
    if Z.max() > 0:
        _max = Z.max()
    else:
        _max = 1e-7
        
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6,
                levels = [_min, 0, _max])
#    else:
#        print('minimum logit is greater than zero')
#        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6,
#                    levels = [-1e-7, 0, Z.max()])
    #%%
    if bool_miss is None:
        # Plot the training points
        ax.scatter(
            X[:, 0], X[:, 1], c=y, cmap=cm_bright, alpha=0.5,edgecolors="k",
            label='complete',**kwargs
        )
    else:
        bool_any = bool_miss.any(1)
        ax.scatter(
            X[~bool_any, 0], X[~bool_any, 1], c=y[~bool_any], cmap=cm_bright, alpha=0.5,edgecolors="k",
            label='complete'
        )
        
        for i, (marker,label) in enumerate(zip(['x','+'],
                                       ['x axis','y axis'])):
            ax.scatter(
                X[bool_miss[:,i], 0], 
                X[bool_miss[:,i], 1], 
                c=y[bool_miss[:,i]], cmap=cm_bright, alpha=0.5,edgecolors="k",
                marker=marker,
                label='partial %s'%label
            )
#        ax.scatter(
#            X[~bool_miss, 0], X[~bool_miss, 1], c=y[~bool_miss], cmap=cm_bright, alpha=0.5,edgecolors="k",
#            label='complete'
#        )
#        ax.scatter(
#            X[bool_miss, 0], X[bool_miss, 1], c=y[bool_miss], cmap=cm_bright, alpha=0.5,edgecolors="k",
#            marker='x',
#            label='partial'
#        )
        
    ax.scatter(
        x_target[:, 0], x_target[:, 1], 
        c=y_target, cmap=cm_bright, 
        edgecolors="k", marker="s",
        label='target'
    )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend(loc="lower left")

    #if ds_cnt == 0:
    #    ax.set_title(name)
#    ax.text(
#        x_max - 0.3,
#        y_min + 0.3,
##        ("%.2f" % score).lstrip("0"),
#        size=15,
#        horizontalalignment="right",
#    )
    
    plt.tight_layout()
    
    return x_min, x_max, y_min, y_max


def draw_grid(model, 
               X,
               x_min = None, x_max = None,
               y_min = None, y_max = None,
               ):    

    if x_min is None:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
    xx, yy, X_grid = _get_grid(X,
                                    x_min, x_max,
                                    y_min, y_max)
    
    
#    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
#    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5        
    fig, axs = plt.subplots(1,2)
    
    for y, ax in zip([0,1], axs):
    
    
        Z = np.concatenate([X_grid,np.full([len(X_grid),1],y)],1)
        
        Z_scaled = (Z - model.scale_mu)/model.scale_std
        
        #Sets the seed
    #    np.random.seed(42)
        #    print('tf seed', seed_model)
    #    tf.random.set_seed(blamm_param['seed_model'])
        
        p_r_x, _ = get_p2(Z_scaled, model, 1,
                          model.idx_input,
                          bool_mcar=0,
                          training = False)
            
        p_r_x = p_r_x.numpy()[:,0]
        p_r_x = p_r_x.reshape(xx.shape)
            
    #    if not np.array_equal(idx_adv_train,idx_mask):
    #        raise TypeError
            
    #    if 0:
    #        fig = helper_draw.draw_2d_long(X_grid, p_r_x, 
    #                                       ratio,
    #                                       names_row,
    #                                       bool_full = 1)
        if p_r_x.shape[1] == 2:
            
    #        names_ratio = ['x_s,x_t']
            
            draw_p_r_x(X_grid, p_r_x, ax
    #                         names_ratio
                             )
                
    #    fig.suptitle(str_1)

#    if p_r_x.shape[1] == 4:
#    fig.set_size_inches( w = 20,h = 10)
#    elif p_r_x.shape[1] == 2:    
    fig.set_size_inches( w = 10,h = 5)
        
    return fig


def draw_p_r_x(X, p_r_x, 
               ax
#               names_row,
#             names_ratio,
#             bool_full = False
             ):
        
#    if not bool_full:
#        raise TypeError
    
#    x1_pts = np.unique(X[:,0])
#    x2_pts = np.unique(X[:,1])
    
    ratio_est_grid = p_r_x[:,0]
    ratio_est_grid = ratio_est_grid.reshape(len(x1_pts),
                                            len(x2_pts))
    draw = ratio_est_grid
    
    center = 0.5
    vmin = 0
    vmax = 1

    title = r'$P_{R\mid X,y;\phi}$' #+'(%s observed)'%'both'

    draw_heat(draw, x1_pts, x2_pts, 
              center=center, 
              vmin=vmin, vmax=vmax, 
              title =title, 
    #          name_0, name_1,
    #          name_modif, 
              annot = False,
              fmt = '%.2f',
              n_skip_label = 10,
#              fig = fig,
              ax = ax,
              bool_size = 0,
              name_0=r'$x_0$',
              name_1=r'$x_1$',
#                  linewidths = 0.01,
#                  linecolor = 'black',
              square = True)
        
#    return fig
    

import helper_draw
def draw_ratio(drawPath2,
               model, X
               ):    

    X_sub = X[:,model.idx_input[:-1]]
    temp = np.stack([X_sub.min(0),
                     X_sub.max(0)])
        
#    x1_pts = np.arange(-8,7.9,0.1)
#    x2_pts = x1_pts
    x1_pts = np.arange(temp[0,0],temp[1,0],0.1)
    x2_pts = np.arange(temp[0,1],temp[1,1],0.1)
    
    from etc import other_favs
    X_grid = other_favs.get_mesh_ravel([x1_pts, x2_pts])
    y = 0
                        
    Z = np.concatenate([X_grid,np.full([len(X_grid),1],y)],1)
        
    Z_scaled = (Z - model.scale_mu)/model.scale_std
        
    p_r_x, _ = get_p2(Z_scaled, model, 1,
                      model.idx_input,
                      bool_mcar=0,
                      training = False)
    p_r_x = p_r_x.numpy()
    
    
    names_mask = np.array([str(idx_i) for idx_i in model.idx_mask])
#    names_mask[modelidx_mask==idx_s] = 'S'
#    names_mask[idx_mask==idx_t] = 'T'
    
#    names_row = get_row_names(idx_mask, 
#                                          names_mask)
    
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
        names_row = ['']
        
        ratio = p_r_x[:,0]
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
    
    fig.suptitle('')

    if p_r_x.shape[1] == 4:
        fig.set_size_inches( w = 20,h = 10)
    elif p_r_x.shape[1] == 2:    
        fig.set_size_inches( w = 10,h = 5)
        
    drawfile = os.path.join(drawPath2,'ratio.png')
    fig.savefig(drawfile, dpi=200, bbox_inches='tight')
    
    return drawfile

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def draw_decision_boundary2(X, y, 
                          x_target, y_target,
                          model, ax,
                          y_in,
                          x_min = None, x_max = None,
                          y_min = None, y_max = None,
                          bool_miss = None):
    
    '''
    Draws the decision boundary of the NN
    '''
    
    if x_min is None:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
    xx, yy, X_grid = _get_grid(X,
                               x_min, x_max,
                               y_min, y_max)
    
    Z_temp = np.concatenate([X_grid,np.full([len(X_grid),1],y_in)],1)
        
    Z_scaled = (Z_temp - model.scale_mu)/model.scale_std
        
    p_r_x, _ = get_p2(Z_scaled, model, 1,
                      model.idx_input,
                      bool_mcar=0,
                      training = False)
    
    p_r_x = p_r_x.numpy()[:,0]
    
#    Z = model(Z_scaled).numpy()
#    Z = logits>0
    Z = p_r_x.reshape(xx.shape)
    
    # ----------------- Plotting -----------------
    #plt.figure(figsize=(10, 8))
    # Create a color map for the predictions
    colors = [(1, 0, 0),  # red at 0.0
              (1, 1, 1),  # white at 0.5
              (0, 0, 1)]  # blue at 1.0
#    cmap = LinearSegmentedColormap.from_list('RedWhiteBlue', colors, N=256)
#    cmap = LinearSegmentedColormap.from_list('RedBlueEndpoints', ['#FFAAAA', '#AAAAFF'], N=256)
    cmap = LinearSegmentedColormap.from_list('RedWhiteBlue', ['#FFAAAA', 'white', '#AAAAFF'], N=256)

    # Set up normalization so that 0.5 is the center of the colormap.
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)


    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
#    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
                                
    # Plot the decision boundary by assigning a color to each point in the mesh.
    contour = ax.contourf(xx, yy, Z, 
#                cmap=cmap_light, alpha=0.6,
#                levels = [0, 0.5, 1]
                levels=50, cmap=cmap, norm=norm
                ) 
#    cbar = plt.colorbar(contour, ax=ax)
#    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
#    cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'])

    #%%
    if bool_miss is None:
        # Plot the training points
        ax.scatter(
            X[y==y_in, 0], X[y==y_in, 1], 
            c=y[y==y_in], 
            cmap=cm_bright, alpha=0.5,edgecolors="k",
            norm=Normalize(vmin=0, vmax=1),
            label='complete'
        )
    else:
        bool_any = bool_miss.any(1)
        ax.scatter(
            X[~bool_any, 0], X[~bool_any, 1], c=y[~bool_any], cmap=cm_bright, alpha=0.5,edgecolors="k",
            label='complete'
        )
        
        for i, (marker,label) in enumerate(zip(['x','+'],
                                       ['x axis','y axis'])):
            ax.scatter(
                X[bool_miss[:,i], 0], 
                X[bool_miss[:,i], 1], 
                c=y[bool_miss[:,i]], cmap=cm_bright, alpha=0.5,edgecolors="k",
                marker=marker,
                label='partial %s'%label
            )
#        ax.scatter(
#            X[~bool_miss, 0], X[~bool_miss, 1], c=y[~bool_miss], cmap=cm_bright, alpha=0.5,edgecolors="k",
#            label='complete'
#        )
#        ax.scatter(
#            X[bool_miss, 0], X[bool_miss, 1], c=y[bool_miss], cmap=cm_bright, alpha=0.5,edgecolors="k",
#            marker='x',
#            label='partial'
#        )
        
    ax.scatter(
        x_target[:, 0], x_target[:, 1], 
        c=y_target, cmap=cm_bright, 
        edgecolors="k", marker="s",
        label='target',
         vmin=0, vmax=1
    )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend(loc="lower left")

    #if ds_cnt == 0:
    #    ax.set_title(name)
#    ax.text(
#        x_max - 0.3,
#        y_min + 0.3,
##        ("%.2f" % score).lstrip("0"),
#        size=15,
#        horizontalalignment="right",
#    )
    
    plt.tight_layout()
    
    return x_min, x_max, y_min, y_max

def draw_decision_boundary_pca(X_0, y, 
                              X_target_0, y_target,
                              model, ax,
                              x_min = None, x_max = None,
                              y_min = None, y_max = None,
                              bool_miss = None,
                              **kwargs):
    
    '''
    Draws the decision boundary of the NN
    '''

    trnsfrmr = PCA(n_components=2,random_state=42)
    trnsfrmr.fit(X_0)

    X = trnsfrmr.transform(X_0)
    X_target = trnsfrmr.transform(X_target_0)
    
    if x_min is None:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
    xx, yy, grid_points_0 = _get_grid_n(X,
                                      x_min, x_max,
                                      y_min, y_max)
    
    grid_points = trnsfrmr.inverse_transform(grid_points_0)
    Z = model(grid_points).numpy()
    Z = Z.reshape(xx.shape)
#    import ipdb;ipdb.set_trace()
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
                                
    if Z.min() < 0:
        _min = Z.min()
    else:
        _min = -1e-7
    
    if Z.max() > 0:
        _max = Z.max()
    else:
        _max = 1e-7
        
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6,
                levels = [_min, 0, _max])
    #%%
    if bool_miss is None:
        # Plot the training points
        ax.scatter(
            X[:, 0], X[:, 1], c=y, cmap=cm_bright, alpha=0.5,edgecolors="k",
            label='complete',**kwargs
        )
    else:
        bool_any = bool_miss.any(1)
        ax.scatter(
            X[~bool_any, 0], X[~bool_any, 1], c=y[~bool_any], cmap=cm_bright, alpha=0.5,edgecolors="k",
            label='complete'
        )
        
        for i, (marker,label) in enumerate(zip(['x','+'],
                                       ['x axis','y axis'])):
            ax.scatter(
                X[bool_miss[:,i], 0], 
                X[bool_miss[:,i], 1], 
                c=y[bool_miss[:,i]], cmap=cm_bright, alpha=0.5,edgecolors="k",
                marker=marker,
                label='partial %s'%label
            )
        
    ax.scatter(
        X_target[:, 0], X_target[:, 1], 
        c=y_target, cmap=cm_bright, 
        edgecolors="k", marker="s",
        label='target',
        vmin=0, vmax=1
    )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend(loc="lower left")

    #if ds_cnt == 0:
    #    ax.set_title(name)
#    ax.text(
#        x_max - 0.3,
#        y_min + 0.3,
##        ("%.2f" % score).lstrip("0"),
#        size=15,
#        horizontalalignment="right",
#    )
    
    plt.tight_layout()
    
    return x_min, x_max, y_min, y_max