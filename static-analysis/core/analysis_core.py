import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from util_core import te_rollout_addnodes, htrees, plot_htrees, auto_threshold, influential_node_ranking, influential_edge_ranking
import csv

# input variables
edge_type = 'UM_TM' # options: ...
pathway_type="summed" # options: summed, greedy, or None
num_roots = 3
root_selection = "edges" # options: "edges", "nodes"


# Limits
vis_lim = 3
dep_lim = 5

# Dataframe of TE network
cascade_df = pd.read_csv('actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('actors.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))
actors_orig = actors
orig_nodes = list(actors_orig.values())

# initialize places to store results
graphs = {}


# Main
if __name__ == "__main__":
    ##### Pathways analysis #####
       
    #Autothreshold
    te_thresh, graph_df1 = auto_threshold(cascade_df, edge_type, 80)
    print(f'te_thresh: {te_thresh}')
    print(graph_df1)
        
    g = nx.from_pandas_edgelist(graph_df1, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())

    # Identify influential nodes
    if(root_selection == 'nodes'):
        root_nodes = influential_node_ranking(g, pulltop=num_roots, node_names=True)
    elif(root_selection =='edges'):
        #TODO Alina, please put your function in here, you'll likely need to pull the first nodes out of the edges you find, or something similar
    
    print(f'root nodes: {root_nodes}')
        
    all_root_dfs, actors = te_rollout_addnodes(in_roots = root_nodes, in_edges_df = graph_df1, max_visits=vis_lim, actors=actors)
            

    # Graph/tree plotting of paths from root
    root_graphs = {}
    for roots, root_df in all_root_dfs.items():
        g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        root_graphs.update({roots:g})
              
    # Generate tree information in for of lists (1 entry per root node)
    xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos = htrees(root_graphs, edge_type, te_thresh, actors, vis_lim, dep_lim, orig_nodes, path=pathway_type)

    ts_figs = plot_htrees(xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos, te_thresh, edge_type)

    # Save resulting tree plots
    print(ts_figs)
    for n, ax in enumerate(ts_figs):
        ax.figure.savefig(f'test_{n}.png')


