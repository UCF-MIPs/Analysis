import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from util import plot_graphs, plot_betweenness_centrality, plot_degree, te_rollout, te_rollout_addnodes, plot_path_lengths, plot_htrees
import csv

# Directories
degree_dir      = 'results/te-network-degree/' 
degree_diff_dir = 'results/te-network-degree-diff/'
centrality_dir  = 'results/te-network-centrality/'
paths_dir       = 'results/paths/'
graphs_dir      = 'results/graphs/'
tree_dir        = 'results/trees/trees-dynamic/'
cascade_dir     = 'results/cascades'

# Thresholds
te_thresh = 0.05 # used for influence across classifications, ex// TM_TM, UM_TM
te_total_thresh = 0.1

# Limits
vis_lim = 5
dep_lim = 10
max_nodes = 101. # To disregard communities (nodes 101 and up)

# Dataframe of TE network (v2/v4/dynamic)
#graph_df = pd.read_csv('data/v2/gephi_actor_te_edges.csv')
#graph_df = pd.read_csv('data/v4/actor_te_edges.csv')
graph_df = pd.read_csv('data/dynamic/actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = graph_df.loc[(graph_df['Target'] > 1.) & (graph_df['Source']<max_nodes) & (graph_df['Target']<max_nodes)]
#print(graph_df.head())

# Dict of actor names (v2/v4/dynamic)
#actor_df = pd.read_csv('data/v2/actors.csv')
#actor_df = pd.read_csv('data/v4/actors.csv')
actor_df = pd.read_csv('data/dynamic/actors.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))
actors_orig = actors
orig_nodes = list(actors_orig.values())

# Capture all edge types
from_edges = ['UF', 'UM', 'TF', 'TM']
to_edges   = ['UF', 'UM', 'TF', 'TM']

edge_types = ['total_te','total_te']
for i in from_edges:
    for j in to_edges:
        edge_types.append(str(i + '_' + j))

# initialize places to store results
graphs = {}
centrality = []
in_deg_centrality = []
out_deg_centrality = []

# Main
if __name__ == "__main__":
    '''
    for edge_type in edge_types:
        # Filter for TE edges above threshold value
        #graph_df1 = graph_df.loc[(graph_df[edge_type] > te_thresh) & (graph_df['Target'] > 1.)& (graph_df['Source']<101.) & (graph_df['Target']<101.)]
        graph_df1 = cascade_df.loc[(graph_df[edge_type] > te_thresh)]
        g = nx.from_pandas_edgelist(graph_df1, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())

        # Collect generated graphs labeled by edge types (UFTM classifications)
        graphs.update({edge_type:g})
        
        # Differentiate between total TE threshold and individual TE thresholds
        if(edge_type == 'total_te'):
            thresh = te_total_thresh
        elif(edge_type != 'total_te'):
            thresh = te_thresh

        # Plotting functions
        plot_degree(g, thresh, edge_type, degree_dir, degree_diff_dir)
        plot_betweenness_centrality(g, thresh, edge_type, centrality_dir)

    '''

    ##### Pathways analysis #####
    #for edge_type in ['TM_TM', 'TM_TM', 'UM_TM', 'total_te']: #duplicate initial to fix blank plot problem...
    for edge_type in edge_types:
        for te_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            #TODO delete the extra communitty exlusion (2nd time it appears)
            cascade_df = graph_df.loc[(graph_df[edge_type] > te_thresh) & \
                                        (graph_df['Target'] > 1.) & (graph_df['Source']<101.) & (graph_df['Target']<101.)]
            
            # root nodes are those identified previously as most influential.
            # In the dynamic v4, these nodes are 12, 84, 23
            root_nodes = [12, 84, 23]
            lengths, all_root_dfs, actors = te_rollout_addnodes(in_roots = root_nodes, in_edges_df = cascade_df, max_visits=vis_lim, actors=actors)
            plot_path_lengths(lengths = lengths, edge_type = edge_type, te_thresh = te_thresh, paths_dir = paths_dir)
            
            # Graph/tree plotting of paths from root
            root_graphs = {}
            for roots, root_df in all_root_dfs.items():
                g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                root_graphs.update({roots:g})
              
            plot_htrees(root_graphs, tree_dir, edge_type, te_thresh, actors, vis_lim, dep_lim, orig_nodes)

            # Have to redo subgraph generation without added nodes for tree viz
            lengths, all_root_dfs = te_rollout(in_roots = root_nodes, in_edges_df = cascade_df, max_visits=vis_lim)
            root_graphs = {}
            for roots, root_df in all_root_dfs.items():
                g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                root_graphs.update({roots:g})
            plot_graphs(root_graphs, graphs_dir, actors = actors_orig, edge_type=edge_type, te_val = te_thresh)
            
