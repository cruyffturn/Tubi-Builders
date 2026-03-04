# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

@plt.FuncFormatter
def fake_log(x, pos):
    '''
    https://stackoverflow.com/questions/48858854/how-to-apply-logarithmic-axis-labels-without-log-scaling-image-matplotlib-imsho
    '''
    'The two args are the value and tick position'
    if x == int(x):
        return r'$10^{%d}$' % (x)
    else:
        return r'$10^{%.1f}$' % (x)
    
def add_counts( ax, y, loc = 'best', fontsize = 'small'):
    
    '''
    
    Add counts to legend.
    In:
        y:  N,
        
    '''
    handles, labels = ax.get_legend_handles_labels()
    
    for i in range( len( labels)):
        
        idx = np.where( y == labels[i])[0]
#        labels[i] = labels[i] + ' (n=' + str( idx.shape[0]) +')'
        handles[i].set_label( labels[i] + ' (n=' + str( idx.shape[0]) +')')
        
#    ax.legend( handles, labels, loc = loc, fontsize = fontsize)
    ax.legend( loc = loc, fontsize = fontsize)
    
def remove_excess_legend( ax, unq_y, loc = 'best', fontsize = 'small'):
    
    '''
    Removes excess legend entries
    '''
    
    handles, labels = ax.get_legend_handles_labels()
        
    idx_inter = np.array( [labels.index(y_i) for y_i in unq_y])
    idx_diff = np.setdiff1d( range( len(handles)), idx_inter)
    
    [handles[idx_temp].remove() for idx_temp in idx_diff]
    ax.legend( loc = loc, fontsize = fontsize)


def get_x_y_min_max( ax):
    
    '''
    Finds the min and max of x and y axis
    
    Out:
        min_max:    L,2,2       # dim 1 corresponds to x and y dim 2 min and max
    '''
    
    min_max = []
    
    for i in range( len( ax.collections)):
        
        cs = ax.collections[i]
        temp = cs.get_offsets()
        
        if temp.shape[0] != 0:
#            cs.set_offset_position('data')
#            val = cs.get_offsets()
#            
#            cs.set_offsets(temp)
            
            val = temp
            min_max.append( np.expand_dims( np.vstack( [np.min( val,0), np.max( val,0)]),0))
    
    min_max = np.concatenate( min_max,0)
    
    return min_max

def draw_thres( ax, thres_draw, label = '', color = None):
    
    if color is None:
        color = '#2ca02c'
    
    min_max_x = get_x_y_min_max( ax)[:,0,:]     #Getting the x coordinates of the scattered points 
    min_x = np.min(min_max_x[:,0])
    max_x = np.max(min_max_x[:,1])
    eps = (max_x-min_x)/8
    #'Decisioun Bound.'
    ax.plot( [min_x-eps,max_x],[thres_draw,thres_draw], '-.', label = label, color = color) #'Dec. Bound.'
#    ax.xticks(list(plt.xticks()[0]) + extraticks)
#    ax.set_yticks( list( ax.get_yticks()) + [thres_draw])
    
def get_colors_from_values(x, cmap):
    
    '''
    In:
        X:  L; or L,
    Out:
        color_L:    L;
    '''
    
    norm = mpl.colors.Normalize()
    norm.autoscale(x)
    color_L = cmap([norm(x_i) for x_i in x])
    
    return color_L

def get_fmt(a):
    
    if a.dtype == float:
        fmt = '%.2f'
    elif a.dtype == int:
        fmt = '%i'
        
    return fmt

def _fmt(x, fmt = '%.2f'):
    
    if fmt[-1] == 'f' and x == 0:
        return '0'
    else:
        return fmt%x