import networkx as nx
import pandas as pd
import numpy as np
import os
from pathlib import Path
from src import generate_edge_types
from src import influential_node_ranking
from src import influential_edge_ranking
from src import add_aggregate_networks

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Use in interface

edge_types = generate_edge_types.generate_edge_types()
edge_types = edge_types + ['T_T', 'U_U', 'U_T', 'T_U', 'TM_*', 'TF_*', 'UM_*', 'UF_*', '*_TM', '*_TF', '*_UM', '*_UF']
node_rank = 'outdegree'
dataset = 'skrip_v7' # options: skrip_v4, skrip_v7, ukr_v3
te_thresh = 0.01

# Data
skrip_v7_te = 'data/Skripal/v7/indv_network/actor_te_edges_df.csv'
skrip_v7_act = 'data/Skripal/v7/indv_network/actors_df.csv'
ukr_v3_te = 'data/Ukraine/v3/dynamic/actor_te_edges_df_2022_01_01_2022_05_01.csv'
ukr_v3_act = 'data/Ukraine/v3/dynamic/actors_df.csv'

te_df_name = f'{dataset}_te'
act_df_name = f'{dataset}_act'
myvars = locals()
te_df_path = myvars[te_df_name]
act_df_path = myvars[act_df_name]

actor_df = pd.read_csv(act_df_path)
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

# Networks
graph_dict = {}

edge_types2 = ['actors'] + edge_types
node_presence_df = pd.DataFrame(columns = edge_types2)
node_presence_df['actors'] = actors.values()
node_presence_df.fillna(value=0, inplace=True)

infl_weights_df = pd.DataFrame(columns = edge_types2)
infl_weights_df['actors'] = actors.values()
infl_weights_df.fillna(value=0, inplace=True)

# Pre-process #TODO fix to include chunking method
graph_df = pd.read_csv(te_df_path)
graph_df = add_aggregate_networks.add_aggr_nets(graph_df)
 
influencer_dict = {}

for edge_type in edge_types:
    graph_dict[edge_type] = {}
        
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
                #iconvert 'dict_items' dtype to float
                for k, v in edge_data[2].items():
                    w = float(v)
                summed_weight += w
            row_index = infl_weights_df.index[infl_weights_df['actors']==node].to_list()
            infl_weights_df.loc[row_index, [edge_type]]=summed_weight
 

    ### NODE RANK ###

    if node_rank == 'outdegree':
        root_nodes = influential_node_ranking.influential_node_ranking_outdegree(g, node_names=False)
    if node_rank == 'bc':
        root_nodes = influential_node_ranking.influential_node_ranking_bc(g, node_names=False)
    
    influencer_dict[edge_type] = root_nodes 


#influencers = list(influencer_dict.values())
#top_infl = []

#for edge_t in influencers:
#    for act in edge_t:
#        top_infl.append(act[0])

#top_infl = list(set(top_infl))

#mask = node_presence_df['actors'].isin(top_infl)
#top_infl_presence_df = node_presence_df[mask]

#mask = infl_weights_df['actors'].isin(top_infl)
#top_infl_weights_df = infl_weights_df[mask]

node_ranks_df = pd.DataFrame.from_dict(influencer_dict)

node_presence_df.to_csv('infl_presence_df.csv')
infl_weights_df.to_csv('infl_weights_df.csv')
node_ranks_df.to_csv('node_ranks_df.csv')


###################################################################################

'''
infl_type_corr = {}

edge_types3 = ['edge_types'] + edge_types
infl_type_corr_df = pd.DataFrame(columns = edge_types)
infl_type_corr_df['edge_types'] = edge_types
infl_type_corr_df.fillna(value=0, inplace=True)


# Node activity comparison
for edge_type_i in edge_types:
    for edge_type_j in edge_types:
        summed_activity_score = 0
        for actor in actors.values():
            print(actor)
            if edge_type_i != edge_type_j:
                row_index = infl_weights_df.index[infl_weights_df['actors']==actor].to_list()
                # Future warning from pandas on using float() on a .loc
                print(infl_weights_df.loc[row_index, edge_type_i])
                print(row_index)
                a1 = infl_weights_df.loc[row_index, edge_type_i].astype('float')
                print(a1)
                a2 = infl_weights_df.loc[row_index, edge_type_j].astype('float')
                print(a2)
                activity_diff = np.abs(a1 - a2)
                print(activity_diff)
                if(activity_diff == 0 and a1 ==0 and a2 == 0):
                    pass
                else:
                    summed_activity_score += 1/activity_diff
            
        infl_type_corr[str(edge_type_i + '__' + edge_type_j)] = summed_activity_score
        

        row_ind = infl_type_corr_df.index[infl_type_corr_df['edge_types']==edge_type_i].to_list()
        infl_type_corr_df.loc[row_ind, [edge_type_j]] = summed_activity_score

print(infl_type_corr)
print(infl_type_corr_df)


infl_type_corr_df.to_csv('infl_type_corr_df.csv')
'''





