import networkx as nx
import pandas as pd
from src import auto_threshold
from src import te_rollout
from src import htrees
from src import plot_htrees
from src import influential_node_ranking
from src import influential_edge_ranking
from src import generate_edge_types
from src import generate_trees


'''
# SINGLE RUN
# input variables
edge_type = 'UF_UM' # options: ...
pathway_type="greedy" # options: summed, greedy, or None

# Dataframe of TE network
cascade_df = pd.read_csv('data/actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('data/actors_v4.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

#te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 80, return_df=True)
#te_thresh = auto_threshold.auto_threshold(cascade_df, edge_type, 80, return_df=False)

te_thresh = 0.1
graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
nx.relabel_nodes(g, actors, copy=False)

generate_trees.generate_tree_plots(g, edge_type, te_thresh)
'''

# MULTIRUN
# input variables
edge_types = generate_edge_types.generate_edge_types()

pathway_selection="greedy" # options: summed, greedy, or None

# Dataframe of TE network
cascade_df = pd.read_csv('data/actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('data/actors_v4.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

for edge_type in edge_types:
    for te_thresh in [0.1, 0.2]:
        graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
        g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        nx.relabel_nodes(g, actors, copy=False)
        generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection)


