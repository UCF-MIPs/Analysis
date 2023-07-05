import numpy as np

def auto_threshold(df, col, max_nodes):
    '''
    Drops columns not of interest
    Finds weight threshold for a network that results in a given number of nodes
    '''
    df = df[['Source', 'Target', col]]
    for thresh in np.round(np.linspace(0. ,1. ,41, endpoint=True), 3):
        df_filtered = df.loc[(df[col] > thresh)]
        #num_nodes = df_filtered['Source'].nunique()
        num_nodes = len(set(zip(df_filtered['Source'],df_filtered['Target'])))
        print(num_nodes)
        if(num_nodes < max_nodes):
            break
    return thresh, df_filtered
