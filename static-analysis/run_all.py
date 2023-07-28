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
      
# Use in interface

pathway_selection="greedy" # options: summed, greedy, or None
#edge_type = 'UM_TM'
edge_types = generate_edge_types.generate_edge_types()
#edge_types = ['UM_TM', 'UF_TM', 'UF_TF', 'UM_TF']
#edge_types = ['TM_UM', 'TM_UF', 'TF_UM', 'TM_UF']
te_threshes = [0.075, 0.1]

for edge_type in edge_types:
    for te_thresh in te_threshes:
        #cascade_df = pd.read_csv('data/Ukraine/Actor_TE_Edges_Ukraine_v1.csv', usecols=['Source', 'Target', edge_type])
        #actor_df = pd.read_csv('data/Ukraine/actors_Ukraine_v1.csv')
        #actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

        cascade_df = pd.read_csv('data/Skripal/actor_te_edges_2018_03_01_2018_05_01.csv', usecols=['Source', 'Target', edge_type])

        cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]
        actor_df = pd.read_csv('data/Skripal/actors_v4.csv')
        actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

        #te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 120, return_df=True)
        graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]

        g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        nx.relabel_nodes(g, actors, copy=False)

        # Can replace 'influential_node_ranking with a single root node as a list
        # for example, root_nodes = ['Ian56789']
        root_nodes = influential_node_ranking.influential_node_ranking_outdegree(g, pulltop=5, node_names=True)

        print(root_nodes)
        #generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name=None)

        generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name='Skripal_trees_outdeg_selc')
        #rnodes, xtrees, xpathways, xstrengths, xcolormap_nodes, xcolormap_edges, xpos = generate_trees.generate_tree_data(g, edge_type, te_thresh, pathway_selection, root_nodes)


    





