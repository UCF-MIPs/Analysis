#TODO make "heatmap" function
# make participation coeff function, move both to src

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# creating plots for the same targets

def plot_heatmap_indegree(inf, df, name, results_dir):
    '''
    inf:             influence type
    df:             dataframe, ex// TM_out_aggr_wdf, or
                    "Trustworthy-Mainstream in edge aggregated weight dataframe"
    results_dir:    string of results dir
    '''
    aggr_type = str('*_' + inf)
    df = df.drop(['actors'], axis=1)
    print(df.head())
    print(df.columns) #TODO make sure this matches yticklabels
    # Data
    heatmap = np.empty((5, len(df[aggr_type])))
    for i, (column_name, column_data) in enumerate(df.items()):
        heatmap[i] = column_data
    # renaming the first label in y-axis
    if inf == 'TM':
        df = df.rename(columns={aggr_type:f'(TM,UM,TF,UF)_{inf}'})
    elif inf == 'UM':
        df = df.rename(columns={aggr_type:f'(UM,TM,UF,TF)_{inf}'})
    elif inf == 'TF':
        df = df.rename(columns={aggr_type:f'(TF,UF,TM,UM)_{inf}'})
    elif inf == 'UF':
        df = df.rename(columns={aggr_type:f'(UF,TF,UM,TM)_{inf}'})
    # Plotting
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
    ax.set_yticks(range(5))
    labels = df.columns
    ax.set_yticklabels(labels)
    ax.set_xlabel('rank of actors')
    ax.set_title(f'{name} {inf} target actors incoming influence')
    cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
    cbar.set_label('Transfer Entropy')
    ax.set_aspect('auto')
    plt.savefig(f'{results_dir}/{name}_{inf}_in_activity.png')

## creating plots for the same sources

def plot_heatmap_outdegree(inf, df, name, results_dir):
    '''
    inf:             source influence type
    df:             dataframe, ex// TM_out_aggr_wdf, or
                    "Trustworthy-Mainstream out edge aggregated weight dataframe"
    results_dir:    string of results dir
    '''
    aggr_type = str(inf + '_*')
    df = df.drop(['actors'], axis=1)
    print(df.head())
    print(df.columns) #TODO make sure this matches yticklabels
    # Data
    heatmap = np.empty((5, len(df[aggr_type])))
    for i, (column_name, column_data) in enumerate(df.items()):
        heatmap[i] = column_data
    # renaming the first label in y-axis
    if inf == 'TM':
        df = df.rename(columns={aggr_type:f'{inf}_(TM,UM,TF,UF)'})
    elif inf == 'UM':
        df = df.rename(columns={aggr_type:f'{inf}_(UM,TM,UF,TF)'})
    elif inf == 'TF':
        df = df.rename(columns={aggr_type:f'{inf}_(TF,UF,TM,UM)'})
    elif inf == 'UF':
        df = df.rename(columns={aggr_type:f'{inf}_(UF,TF,UM,TM)'})
    # Plotting
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
    ax.set_yticks(range(5))
    labels = df.columns
    ax.set_yticklabels(labels)
    ax.set_xlabel('rank of actors')
    ax.set_title(f'{name} {inf} source actors outgoing influence')
    cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
    cbar.set_label('Transfer Entropy')
    ax.set_aspect('auto')
    plt.savefig(f'{results_dir}/{name}_{inf}_out_activity.png')

