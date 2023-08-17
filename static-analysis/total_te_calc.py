import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
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
te_threshes = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
datasets = ['skrip_v4', 'skrip_v7', 'ukr_v1', 'ukr_v3'] # options: skrip_v4, skrip_v7, ukr_v3


summed_df = pd.DataFrame(edge_types, columns = ['Influence_type'])


for dataset in datasets:

    # Data
    skrip_v4_te = 'data/Skripal/v4/actor_te_edges_2018_03_01_2018_05_01.csv'
    skrip_v4_act = 'data/Skripal/v4/actors_v4.csv'
    # note for skrip v3, need to filter last 10 actors (platforms) like so
    #cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & \
    #(cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & \
    #(cascade_df['Target']<101.)]

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

    for te_thresh in te_threshes:
        summed_te = {}
        for edge_type in edge_types:
        
            graph_df = pd.DataFrame() 
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

            print(graph_df)
            summed_te[edge_type] = sum(graph_df[edge_type])
   
        df = pd.Series(summed_te).to_frame().reset_index()
        # Rename columns
        df.columns = ['Influence_type', f'{dataset}_{te_thresh}_summed_TE'] 
        summed_df = summed_df.merge(df, how='left', on='Influence_type')


summed_df.to_csv(f'summed_te.csv')    
