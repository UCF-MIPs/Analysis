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
#edge_types = generate_edge_types.generate_edge_types()
#pathway_selection="greedy" # options: summed, greedy, or None

# Dataframe of TE network
#Skripal
#cascade_df = pd.read_csv('data/Skripal/actor_te_edges_2018_03_01_2018_05_01.csv')
#cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]
#actor_df = pd.read_csv('data/Skripal/actors_v4.csv')
#actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

#Ukraine
#cascade_df = pd.read_csv('data/Ukraine/Actor_TE_Edges_Ukraine_v1.csv')
#actor_df = pd.read_csv('data/Ukraine/actors_Ukraine_v1.csv')
#actors = dict(zip(actor_df.actor_id, actor_df.actor_label))


'''
# Iterating through thresholds
for edge_type in edge_types:
    for te_thresh in [0.2, 0.25, 0.3]:
        graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
        g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        nx.relabel_nodes(g, actors, copy=False)
      
        # graphs
        nx.draw(g, pos=nx.circular_layout(g), with_labels=True)
        plt.savefig(f'graphs/{edge_type}_{te_thresh}.png')
        plt.clf()

        root_nodes = influential_node_ranking.influential_node_ranking(g, pulltop=5, node_names=True)
        generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name='Ukraine_trees')
        plt.clf()


# Autothresholding 
dataset = 'Ukr'

for edge_type in edge_types:
        
    te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 120, return_df=True)
    #graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
    g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
    nx.relabel_nodes(g, actors, copy=False)
      
    # graphs
    nx.draw(g, pos=nx.circular_layout(g), with_labels=True)
    plt.savefig(f'graphs/{dataset}_{edge_type}_{te_thresh}.png')
    plt.clf()

    root_nodes = influential_node_ranking.influential_node_ranking(g, pulltop=5, node_names=True)
    generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name='Ukraine_trees')
    plt.clf()
''' 
       
# Use in interface

pathway_selection="greedy" # options: summed, greedy, or None
edge_type = 'UM_TM'
te_thresh = 0.1

cascade_df = pd.read_csv('data/Ukraine/Actor_TE_Edges_Ukraine_v1.csv', usecols=edge_type)
actor_df = pd.read_csv('data/Ukraine/actors_Ukraine_v1.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

#te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 120, return_df=True)
graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]

g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
nx.relabel_nodes(g, actors, copy=False)

# Can replace 'influential_node_ranking with a single root node as a list
# for example, root_nodes = ['Ian56789']
root_nodes = influential_node_ranking.influential_node_ranking(g, pulltop=5, node_names=True)

generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name='trees')
 


