import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
from src import auto_threshold
from src import te_rollout
from src import htrees
from src import plot_htrees
from src import influential_node_ranking
from src import influential_edge_ranking
from src import generate_edge_types
from src import generate_trees

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)


# Use in interface

pathway_selection="greedy" # options: summed, greedy, or None
#edge_types = ['TF_TM']
edge_types = generate_edge_types.generate_edge_types()
node_rank = 'outdegree'
#node_rank = 'manual'
#manual_root = ['BombshellDAILY']
dataset = 'skrip_v7' # options: skrip_v4, skrip_v7, ukr_v3
te_thresh = 0.02

# Data
skrip_v4_te = 'data/Skripal/v4/indv_network/actor_te_edges_2018_03_01_2018_05_01.csv'
skrip_v4_act = 'data/Skripal/v4/indv_network/actors_v4.csv'
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

actor_df = pd.read_csv(act_df_path)
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

# Results dir
if node_rank == 'manual':
    dir_name = f'results/{dataset}_{node_rank}_{manual_root}_{pathway_selection}'
else:
    dir_name = f'results/{dataset}_{node_rank}_{pathway_selection}'

path = Path(dir_name)
path.mkdir(parents=True, exist_ok=True)

# Networks
graph_dict = {}

edge_types2 = ['actors'] + edge_types
node_presence_df = pd.DataFrame(columns = edge_types2)
node_presence_df['actors'] = actors
node_presence_df.fillna(value=0, inplace=True)

infl_weights_df = pd.DataFrame(columns = edge_types2)
infl_weights_df['actors'] = actors
infl_weights_df.fillna(value=0, inplace=True)


# Save node influence ranking for each type (top ~20)
influencer_dict = {}

for edge_type in edge_types:
    graph_dict[edge_type] = {}
        
    ### CREATING TE NETWORK ###
        
    #te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 120, return_df=True) 
    graph_df = pd.DataFrame() 
    for chunk in pd.read_csv(te_df_path, chunksize=1000, usecols=['Source', 'Target', edge_type]):
        new_chunk = chunk[chunk[edge_type] > te_thresh]
        graph_df = pd.concat([graph_df, new_chunk])
            
    #print(graph_df)

    g = nx.from_pandas_edgelist(graph_df, source='Source', target='Target', edge_attr=[edge_type], create_using=nx.DiGraph())
    nx.relabel_nodes(g, actors, copy=False)

    graph_dict[edge_type] = g

    # identify which influence types nodes appear in
    for node in actors.values():
        ## presence df filling
        if g.has_node(node):
            row_index = node_presence_df.index[node_presence_df['actors']==node].to_list()
            node_presence_df.loc[row_index, [edge_type]] = 1

        ## weight df filling
        if g.has_node(node):
            out_edges = g.out_edges(node, data=True)
            summed_weight = 0
            for edge_data in out_edges:
                #convert 'dict_items' dtype to float
                for k, v in edge_data[2].items():
                    w = float(v)
                summed_weight += w
            row_index = infl_weights_df.index[infl_weights_df['actors']==node].to_list()
            infl_weights_df.loc[row_index, [edge_type]]=summed_weight
 

    ### NODE RANK ###

    if node_rank == 'outdegree':
        root_nodes = influential_node_ranking.influential_node_ranking_outdegree(g, pulltop=20, node_names=False)
    if node_rank == 'bc':
        root_nodes = influential_node_ranking.influential_node_ranking_bc(g, pulltop=20, node_names=False)
    if node_rank == 'manual':
        root_nodes = manual_root
    
    influencer_dict[edge_type] = root_nodes 


influencers = list(influencer_dict.values())
top_infl = []

for edge_t in influencers:
    for act in edge_t:
        top_infl.append(act[0])

top_infl = list(set(top_infl))

mask = node_presence_df['actors'].isin(top_infl)
top_infl_presence_df = node_presence_df[mask]

mask = infl_weights_df['actors'].isin(top_infl)
top_infl_weights_df = infl_weights_df[mask]


node_ranks_df = pd.DataFrame.from_dict(influencer_dict)

# all node data in node_presence_df and infl_weights_df, ranking in node_ranks_df
# top influential node data in top_infl_presence_df and top_infl_weights_df

print(top_infl_presence_df)
print(top_infl_weights_df)
print(node_ranks_df)

top_infl_presence_df.to_csv('top_infl_presence_df.csv')
top_infl_weights_df.to_csv('top_infl_weights_df.csv')
node_ranks_df.to_csv('node_ranks_df.csv')


