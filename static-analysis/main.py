import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from src import auto_threshold
from src import te_rollout
from src import htrees
from src import plot_htrees
from src import influential_node_ranking
from src import influential_edge_ranking
from src import generate_edge_types
from src import generate_trees

#from src import temp # my addition

      
# Use in interface

pathway_selection="summed"#"greedy" # options: summed, greedy, or None
edge_type = 'TM_TM'#'UM_TM'
te_thresh = 0.25 #0.1

cascade_df = pd.read_csv('data/Ukraine/Actor_TE_Edges_Ukraine_v1.csv', usecols=['Source', 'Target', edge_type])
actor_df = pd.read_csv('data/Ukraine/actors_Ukraine_v1.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

#te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 120, return_df=True)
graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]

g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
nx.relabel_nodes(g, actors, copy=False)

# Can replace 'influential_node_ranking with a single root node as a list
# for example, root_nodes = ['Ian56789']
root_nodes = influential_node_ranking.influential_node_ranking(g, pulltop=5, node_names=True)

generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name=None)