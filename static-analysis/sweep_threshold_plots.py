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

#Skripal
actor_df = pd.read_csv('data/Skripal/v4/actors_v4.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

#for edge_type in edge_types:
#    cascade_df = pd.read_csv('data/Skripal/actor_te_edges_2018_03_01_2018_05_01.csv', usecols=['Source', 'Target', edge_type])
#    sweep_threshold_plots.plot_sweep(te_threshes, cascade_df, edge_type, 'outdegree')
    

csv_name = 'data/Skripal/v4/actor_te_edges_2018_03_01_2018_05_01.csv'
#plots = sweep_threshold_plots.plot_all_sweep(te_threshes, edge_types, 'outdegree', csv_name)

graph_dict = {}
for edge_type in edge_types:
    iter_csv = pd.read_csv(csv_name, iterator=True, chunksize=1000, usecols=['Source', 'Target', edge_type])
    graph_df = pd.concat([chunk[chunk[edge_type] > 0] for chunk in iter_csv])
    #g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
    graph_dict[edge_type] = graph_df

print(graph_dict)
#plot_quadrant.plot_quadrant(graph_dict, 'sweep-outdegree')

sweep_threshold_plots.plot_all_sweep(te_threshes, edge_types, 'outdegree', csv_name)
#sweep_threshold_plots.plot_all_sweep(te_threshes, edge_types, 'bc', csv_name)
#sweep_threshold_plots.plot_all_sweep(te_threshes, edge_types, 'num_nodes', csv_name)
#sweep_threshold_plots.plot_all_sweep(te_threshes, edge_types, 'num_edges', csv_name)



