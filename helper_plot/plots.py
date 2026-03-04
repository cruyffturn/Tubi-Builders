# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#import matplotlib.transforms as mtrans

from .helper_plot import get_fmt, _fmt


def draw_heat(Z, param_0, param_1, 
              center, vmin, vmax, title, 
              name_0 = '', name_1 = '',
              name_modif = '',
              annot = None,
              fmt='%.2g',
              fig = None,
              ax = None,
              n_skip_label = None,
              bool_size = 1,
              **kwargs):
    
    '''
    Draw Z[0,0] is the left top of the image
    
    '''
    
    return_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots()
        
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
                
#    title += '%.2f'%q
    
    
    fmt_x = get_fmt(param_1)
    x_draw = [fmt_x%x_i for x_i in param_1]
    
    fmt_y = get_fmt(param_0)
    y_draw = np.flip(param_0)
    y_draw = [fmt_y%x_i for x_i in y_draw]

    if n_skip_label is not None:    
        for i, _ in enumerate(x_draw):
            if i % n_skip_label != 0:
                x_draw[i] = ''
                
        for i, _ in enumerate(y_draw):
            if i % n_skip_label != 0:
                y_draw[i] = ''
    
#    import ipdb;ipdb.set_trace()
    Z_draw = np.flip(Z,0)    
        
    if type(annot) is not bool:
        annot = np.flip(annot,0)
        fmt = ''
    else:
        if annot:
            annot = np.vectorize(lambda x:_fmt(x, fmt))(Z_draw)
            fmt = ''
        
    sns.heatmap(Z_draw, 
                xticklabels=x_draw, 
                yticklabels=y_draw,
                ax = ax,
                cmap=cmap, 
                center= center,
                vmin = vmin,
                vmax = vmax,
                annot = annot,
                fmt = fmt,
                **kwargs
                )
    
    ax.set_title(title)
    ax.set_xlabel(name_1)
    ax.set_ylabel(name_0)    

    
    if bool_size:
        fig.set_size_inches(24,10)
    
    if name_modif != '':
        fig.suptitle(name_modif, y=0.95)
        
    if return_ax:
        return fig, ax
