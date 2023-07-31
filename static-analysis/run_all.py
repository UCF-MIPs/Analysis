import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from src import auto_threshold
from src import te_rollout
from src import htrees
from src import plot_htrees
from src import influential_node_ranking
from src import influential_edge_ranking
from src import generate_edge_types
from src import generate_trees
      
# Use in interface

pathway_selection="greedy" # options: summed, greedy, or None
#edge_type = 'UM_TM'
edge_types = generate_edge_types.generate_edge_types()
te_threshes = [0.075, 0.1]
node_rank = 'outdegree'
dataset = 'ukr_v3' # options: skrip_v4, skrip_v7, ukr_v3

# Data
skrip_v4_te = 'data/Skripal/v4/indv_network/actor_te_edges_2018_03_01_2018_05_01.csv'
skrip_v4_act = 'data/Skripal/v4/indv_network/actors_v4.csv'
# note for skrip v3, need to filter last 10 actors (platforms) like so
#cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

skrip_v7_te = 'data/Skripal/v7/indv_network/actor_te_edges_df.csv'
skrip_v7_act = 'data/Skripal/v7/indv_network/actors_df.csv'

ukr_v1_te = 'data/Ukraine/v1/Actor_TE_Edges_Ukraine_v1.csv'
ukr_v1_act = 'data/Ukraine/v1/actors_Ukraine_v1.csv'

ukr_v3_te = 'data/Ukraine/v3/dynamic/actor_te_edges_df_2022_01_01_2022_05_01.csv'
ukr_v3_act = 'data/Ukraine/v3/dynamic/actors_df.csv'

te_df_name = f'{dataset}_te'
act_df_name = f'{dataset}_act'
myvars = locals()
te_df_path = myvars[te_df_name]
act_df_path = myvars[act_df_name]

# Results dir
dir_name = f'results/{dataset}_{node_rank}_{pathway_selection}'
path = Path(dir_name)
path.mkdir(parents=True, exist_ok=True)

for edge_type in edge_types:
    for te_thresh in te_threshes:
        
        cascade_df = pd.read_csv(te_df_path, usecols=['Source', 'Target', edge_type])
        actor_df = pd.read_csv(act_df_path)
        actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

        #te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 120, return_df=True)
        graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]

        g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        nx.relabel_nodes(g, actors, copy=False)

        # Can replace 'influential_node_ranking with a single root node as a list
        # for example, root_nodes = ['Ian56789'] 
        if node_rank == 'outdegree':
            root_nodes = influential_node_ranking.influential_node_ranking_outdegree(g, pulltop=5, node_names=True)
        if node_rank == 'bc':
            root_nodes = influential_node_ranking.influential_node_ranking_bc(g, pulltop=5, node_names=True)

        #generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name=None)
        generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name=dir_name)
        #rnodes, xtrees, xpathways, xstrengths, xcolormap_nodes, xcolormap_edges, xpos = generate_trees.generate_tree_data(g, edge_type, te_thresh, pathway_selection, root_nodes)


