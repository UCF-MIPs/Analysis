import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from src import generate_edge_types
from src import add_aggregate_networks

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

edge_types = generate_edge_types.generate_edge_types()
edge_types = edge_types + ['TM_*', 'TF_*', 'UM_*', 'UF_*', '*_TM', '*_TF', '*_UM', '*_UF']

dataset = 'skrip_v7' # options: skrip_v7, ukr_v3

# TODO fix? not sure if *_* == total_te
if dataset=='ukr_v3':
    # rename total_te to *_*
    dict1 = {"total_te":"*_*"}
    edge_types = [dict1.get(n,n) for n in edge_types]

# Data
skrip_v7_te = 'multilayer-network/Data/actor_te_edges_df - Skripal.csv'
skrip_v7_act = 'multilayer-network/Data/actors_df - Skripal.csv'
ukr_v3_te = 'Data/actor_te_edges_df_2022_01_01_2022_05_01 - Ukraine.csv'
ukr_v3_act = 'Data/actors_df - Ukraine.csv'

te_df_name = f'{dataset}_te'
act_df_name = f'{dataset}_act'
myvars = locals()
te_df_path = myvars[te_df_name]
act_df_path = myvars[act_df_name]

actor_df = pd.read_csv(act_df_path)
if dataset=='ukr_v3':
    actors = dict(zip(actor_df.actor_id, actor_df.user_id)) # For just indv users
elif dataset=='skrip_v7':
    actors = dict(zip(actor_df.actor_id, actor_df.actor_label)) # For "full_actors" file

# Networks
graph_dict = {}

edge_types2 = ['actors'] + edge_types

out_infl_weights_df = pd.DataFrame(columns = edge_types2)
out_infl_weights_df['actors'] = actors.values()
#out_infl_weights_df.fillna(value=0, inplace=True)

in_infl_weights_df = pd.DataFrame(columns = edge_types2)
in_infl_weights_df['actors'] = actors.values()
#in_infl_weights_df.fillna(value=0, inplace=True)

 
'''
# Drop rows with all 0's
out_infl_weights_df = out_infl_weights_df.loc[~( \
        (out_infl_weights_df['UM_*'] == 0) & \
        (out_infl_weights_df['UF_*'] == 0) & \
        (out_infl_weights_df['TM_*'] == 0) & \
        (out_infl_weights_df['TF_*'] == 0) \
        )]

in_infl_weights_df = in_infl_weights_df.loc[~( \
        (in_infl_weights_df['UM_*'] == 0) & \
        (in_infl_weights_df['UF_*'] == 0) & \
        (in_infl_weights_df['TM_*'] == 0) & \
        (in_infl_weights_df['TF_*'] == 0) \
        )]
'''





# Pre-process #TODO fix to include chunking method
graph_df = pd.read_csv(te_df_path)

# Check number of unique users
num_us = len(pd.unique(graph_df[['Source', 'Target']].values.ravel('K')))
print(f'number of unique users: {num_us}')
graph_df = add_aggregate_networks.add_aggr_nets(graph_df)

for edge_type in edge_types:
    graph_dict[edge_type] = {}
        
    g = nx.from_pandas_edgelist(graph_df, source='Source', target='Target', edge_attr=[edge_type], create_using=nx.DiGraph())
    nx.relabel_nodes(g, actors, copy=False)

    graph_dict[edge_type] = g

    # identify which influence types nodes appear in
    for node in actors.values():

        ## out weight df filling
        if g.has_node(node):
            out_edges = g.out_edges(node, data=True)
            summed_weight = 0
            for edge_data in out_edges:
                #convert 'dict_items' dtype to float
                for k, v in edge_data[2].items():
                    w = float(v)
                summed_weight += w
            row_index = out_infl_weights_df.index[out_infl_weights_df['actors']==node].to_list()
            out_infl_weights_df.loc[row_index, [edge_type]]=summed_weight

        ## in weight df filling
        if g.has_node(node):
            in_edges = g.in_edges(node, data=True)
            summed_weight = 0
            for edge_data in in_edges:
                #iconvert 'dict_items' dtype to float
                for k, v in edge_data[2].items():
                    w = float(v)
                summed_weight += w
            row_index = in_infl_weights_df.index[in_infl_weights_df['actors']==node].to_list()
            in_infl_weights_df.loc[row_index, [edge_type]]=summed_weight

out_infl_weights_df.to_csv(f'multilayer-network/Data/preprocessed/{dataset}_out_infl_weights_df.csv')
in_infl_weights_df.to_csv(f'multilayer-network/Data/preprocessed/{dataset}_in_infl_weights_df.csv')

