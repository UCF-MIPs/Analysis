import numpy as np
from src.htrees import htree 


def auto_threshold(df, col, max_nodes, return_df = False):
    '''
    Drops columns not of interest
    Finds weight threshold for a network that results in a given number of nodes
    '''
    df = df[['Source', 'Target', col]]
    for thresh in np.round(np.linspace(0. ,1. ,41, endpoint=True), 3):
        df_filtered = df.loc[(df[col] >= thresh)]
        num_nodes = len(set(zip(df_filtered['Source'],df_filtered['Target'])))
        print(num_nodes)
        if(num_nodes < max_nodes):
            break
    if return_df==True:
        return thresh, df_filtered
    elif return_df==False:
        return thresh


def auto_threshold_tree(df, col, root, max_nodes, return_tree = False):
    '''
    Finds an appropriate threshold to generate a tree under the given max_nodes value.
    '''
    df = df[['Source', 'Target', col]]
    for thresh in np.round(np.linspace(0. ,1. ,41, endpoint=True), 3):
        df_filtered = df.loc[(df[col] >= thresh)]
        tree, pathway, strength, colormap_nodes, colormap_edges, pos = htrees.htree(graph, \
                edge_type, thresh, visited_lim, depth_lim, orig_nodes, path=None)
        num_nodes = len(tree.nodes)
        print(num_nodes)
        if(num_nodes < max_nodes):
            break
    if return_tree==True:
        return thresh, tree, pathway, strength, colormap_nodes, colormap_edges, pos
    elif return_tree==False:
        return thresh




