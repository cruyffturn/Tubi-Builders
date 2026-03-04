# -*- coding: utf-8 -*-
import os
import re

def get_tex(df,index=0):    
    '''
    https://stackoverflow.com/questions/55216709/is-there-any-way-to-add-borders-to-a-table-generated-by-to-latex
    '''
    
    tex_content = df.to_latex(index=index, escape=True)
#    print(tex_content)
    re_borders = re.compile(r"begin\{tabular\}\{([^\}]+)\}")
    borders = re_borders.findall(tex_content)[0]
    borders = '|'.join(list(borders))
    tex_content = re_borders.sub("begin{tabular}{" + borders + "}", tex_content)
#    print(tex_content)
    
    return tex_content

def _save_tex(path, name, tex_content):
    
    if not os.path.exists(os.path.join(path,'tex')):
        os.makedirs( os.path.join(path,'tex'))
    
    with open(os.path.join(path,'tex','%s.tex'%name), 'w') as tf:
        tf.write(tex_content)
        
def save_tex(path, name, df,
             index=0):
    
    tex_content = get_tex(df,index)
    
    if not os.path.exists(os.path.join(path,'tex')):
        os.makedirs( os.path.join(path,'tex'))
    
    with open(os.path.join(path,'tex','%s.tex'%name), 'w') as tf:
        tf.write(tex_content)