#TODO make "heatmap" function
# make participation coeff function, move both to src

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_heatmap(sc,df, name, results_dir, outgoing=True):
    '''
    sc:             source influence type
    df:             dataframe, ex// TM_out_aggr_wdf, or
                    "Trustworthy-Mainstream out edge aggregated weight dataframe"
    results_dir:    string of results dir
    outgoing:       bool, sets name if using out/in edge df
    '''
    aggr_type = str(sc + '_*')
    df = df.drop(['actors'], axis=1)
    print(df.head())
    print(df.columns) #TODO make sure this matches yticklabels
    # Data
    heatmap = np.empty((5, len(df[aggr_type])))
    for i, (column_name, column_data) in enumerate(df.items()):
        heatmap[i] = column_data
    # Plotting
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
    ax.set_yticks(range(5))
    labels = df.columns
    ax.set_yticklabels(labels)
    ax.set_xlabel('rank of actors')
    ax.set_title(f'{name} TM source actor outgoing activity')
    cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
    cbar.set_label('Transfer Entropy')
    ax.set_aspect('auto')
    if outgoing == True:
        plt.savefig(f'{results_dir}/{name}_{sc}_out_activity.png')
    else:
        plt.savefig(f'{results_dir}/{name}_{sc}_in_activity.png')

