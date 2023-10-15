import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src import auto_threshold
from src import te_rollout
from src import htrees
from src import plot_htrees
from src import influential_node_ranking
from src import influential_edge_ranking
from src import generate_edge_types
from src import generate_trees
from src import sweep_threshold_plots
#from src import plot_quadrant

edge_types = generate_edge_types.generate_edge_types()
#edge_type = 'UM_TM'
pathway_selection="greedy" # options: summed, greedy, or None
te_threshes = np.round(np.linspace(0, 1.0, 100, endpoint=True), 2) 
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

graph_dict = {}
for edge_type in edge_types:
    
    for chunk in pd.read_csv(te_df_path, chunksize=10000, \
    usecols=['Source', 'Target', edge_type]):
        # for skrip v3
        if(dataset == 'skrip_v4'):
            new_chunk = chunk[(chunk[edge_type] > te_thresh) & \
                                (chunk['Source'] > 1.) & \
                                (chunk['Target'] > 1.) & \
                                (chunk['Source'] < 101.) & \
                                (chunk['Target'] < 101.)]
        else:
            new_chunk = chunk[chunk[edge_type] > te_thresh]
        graph_df = pd.concat([graph_df, new_chunk])
    graph_dict[edge_type] = graph_df

print(graph_dict)
#plot_quadrant.plot_quadrant(graph_dict, 'sweep-outdegree')

sweep_threshold_plots.plot_all_sweep(te_threshes, edge_types, 'outdegree', csv_name)
#sweep_threshold_plots.plot_all_sweep(te_threshes, edge_types, 'bc', csv_name)
#sweep_threshold_plots.plot_all_sweep(te_threshes, edge_types, 'num_nodes', csv_name)
#sweep_threshold_plots.plot_all_sweep(te_threshes, edge_types, 'num_edges', csv_name)



