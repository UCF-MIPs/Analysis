import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from util_core import te_rollout_addnodes, htrees, plot_htrees, auto_threshold, influential_node_ranking
import csv

# Limits
vis_lim = 5
dep_lim = 10

# Dataframe of TE network
cascade_df = pd.read_csv('actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = cascade_df.loc[(cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('actors.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))
actors_orig = actors
orig_nodes = list(actors_orig.values())

# Capture all edge types
from_edges = ['UF', 'UM', 'TF', 'TM']
to_edges   = ['UF', 'UM', 'TF', 'TM']

#edge_types = ['UF_TM','UF_TM'] # repeat a few due to blank output #TODO fix
edge_types = []

for i in from_edges:
    for j in to_edges:
        edge_types.append(f"{i}_{j}")

# initialize places to store results
graphs = {}

# Main
if __name__ == "__main__":
    ##### Pathways analysis #####
    for edge_type in edge_types:
       
        #Autothreshold
        te_thresh, graph_df1 = auto_threshold(cascade_df, edge_type, 50)
        print(f'te_thresh: {te_thresh}')
        print(graph_df1)
        
        g = nx.from_pandas_edgelist(graph_df1, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())

        # Identify influential nodes
        root_nodes=influential_node_ranking(g, pulltop=3, node_names=True)
        print(f'root nodes: {root_nodes}')
        #all_root_dfs, actors = te_rollout_addnodes(in_roots = root_nodes, in_edges_df = cascade_df, max_visits=vis_lim, actors=actors)
            
        # Graph/tree plotting of paths from root
        #root_graphs = {}
        #for roots, root_df in all_root_dfs.items():
            #g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
            #root_graphs.update({roots:g})
              
        #plot_htrees(root_graphs, tree_dir, edge_type, te_thresh, actors, vis_lim, dep_lim, orig_nodes, path="summed")
        #htrees(root_graphs, edge_type, te_thresh, actors, vis_lim, dep_lim, orig_nodes, path="summed")

        # Have to redo subgraph generation without added nodes for tree viz, te_rollout vs te_rollout_addnodes
        #lengths, all_root_dfs = te_rollout(in_roots = root_nodes, in_edges_df = cascade_df, max_visits=vis_lim)
        #root_graphs = {}
        #for roots, root_df in all_root_dfs.items():
            #g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
            #root_graphs.update({roots:g})
            
