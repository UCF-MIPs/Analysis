import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#import te_rollout_addnodes, htrees, plot_htrees, auto_threshold, influential_node_ranking, influential_edge_ranking, strongest_path_greedy, strongest_path_summed
#from src import *
from src import auto_threshold
from src import te_rollout_addnodes
from src import htrees
from src import plot_htrees
from src import influential_node_ranking
from src import influential_edge_ranking
from src import generate_edge_types
import csv

# Divide each influence type network by max 
# All threshold from optimal up to 1


# input variables
#edge_type = 'UM_TM' # options: ...
pathway_type="greedy" # options: summed, greedy, or None
num_roots = 5
root_selection = "nodes" # options: "edges", "nodes"

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

edge_types = generate_edge_types.generate_edge_types()

# Main
if __name__ == "__main__":
    ##### Pathways analysis #####
       
    for edge_type in edge_types:

        # TE normalization
        cascade_df[edge_type] = cascade_df[edge_type]/cascade_df[edge_type].max()

        #g = nx.from_pandas_edgelist(graph_df1, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        g = nx.from_pandas_edgelist(cascade_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())

        # Identify influential nodes
        if(root_selection == 'nodes'):
            root_nodes = influential_node_ranking.influential_node_ranking(g, pulltop=num_roots, node_names=True)
            print(f'root nodes: {root_nodes}')
        elif(root_selection =='edges'):
            root_edges, root_nodes, node_centralities = influential_edge_ranking.influential_edge_ranking(g, pulltop=num_roots, edge_names = True)
            print(f'root edges and their centralities: {root_edges}') # list of top root edges
            print(f'root nodes: {root_nodes}') # list of top nodes
            print(f'nodes and their centralities: {node_centralities}') # dictionary of top nodes and corresponding centralities
         
        #Autothreshold
        te_thresh, graph_df1 = auto_threshold.auto_threshold(cascade_df, edge_type, 80)
        #print(f'te_thresh: {te_thresh}')
        #print(graph_df1)
        
        te_threshes = [te_thresh]
        n = te_thresh
        while (te_thresh+n) <=1:
            te_threshes.append(round(te_thresh+n,2))
            n+=0.05

   

        for te_thresh in te_threshes:
            graph_df1 = cascade_df.loc[(cascade_df[edge_type]>te_thresh)]

            g = nx.from_pandas_edgelist(graph_df1, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())

            all_root_dfs, actors = te_rollout_addnodes.te_rollout_addnodes(in_roots = root_nodes, in_edges_df = graph_df1, max_visits=vis_lim, actors=actors)
            

            # Graph/tree plotting of paths from root
            root_graphs = {}
            for roots, root_df in all_root_dfs.items():
                g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                root_graphs.update({roots:g})
              
            # Generate tree information in for of lists (1 entry per root node)
            rnodes, xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos = htrees.htrees(root_graphs, edge_type, te_thresh, actors, vis_lim, dep_lim, orig_nodes, path=pathway_type)

            ts_figs = plot_htrees.plot_htrees(xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos, te_thresh, edge_type)


            #TODO fix names in case of spaces
            #actors2 = pd.DataFrame(actors)
            #actors2.columns = actors2.columns.str.replace(' ', '')

            # Save resulting tree plots
            print(ts_figs)
            for ax, root in zip(ts_figs, rnodes):
                print(root)
                # check to see if root node is mapped somewhere
                ax.figure.savefig(f'{edge_type}_te_thresh{te_thresh}_root{root}.png')
        


