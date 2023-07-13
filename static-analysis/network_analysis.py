import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from src import generate_edge_types, all_betweenness_centrality, in_out_degrees
import numpy as np

# Dataframe of TE network
cascade_df = pd.read_csv('data/actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('data/actors_v4.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

edge_types = generate_edge_types.generate_edge_types()

bc_node_values = {}
bc_edge_values = {}

for edge_type in edge_types:
    print('Edge Type: ', edge_type)
    te_thresh = 0.1
    graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
    g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())

    ########### Calculating all betwenness centrality values and outputing distributions ###########
    bc_node_list, bc_edge_list = all_betweenness_centrality.all_betweenness_centrality(g)

    bc_node_mean = np.mean(bc_node_list)
    bc_node_std = np.std(bc_node_list)
    bc_edge_mean = np.mean(bc_edge_list)
    bc_edge_std = np.std(bc_edge_list)

    bc_node_values[edge_type] = bc_node_list
    bc_edge_values[edge_type] = bc_edge_list

    #print(bc_node_values)
    #print(bc_edge_values)

    # Create distribution plots for node betweenness centrality
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.hist(bc_node_list, bins=20)
    plt.title(edge_type + ' Node Betweenness Centrality Distribution')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Frequency')
    plt.text(0.8, 0.9, f'Mean: {bc_node_mean:.2f}\nStd: {bc_node_std:.2f}', transform=plt.gca().transAxes)
    
    # Create distribution plots for edge betweenness centrality
    plt.subplot(1, 2, 2)
    plt.hist(bc_edge_list, bins=20)
    plt.title(edge_type + ' Edge Betweenness Centrality Distribution')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Frequency')
    plt.text(0.8, 0.9, f'Mean: {bc_edge_mean:.2f}\nStd: {bc_edge_std:.2f}', transform=plt.gca().transAxes)
    fig_name = edge_type + '_bc_distribution.png'
    plt.savefig('./bc_distributions/' + fig_name)
    
    plt.tight_layout()
    #plt.show()

    ########### Calculating average in-degree and out-degree all betwenness centrality values and outputing distributions ###########

    degrees = in_out_degrees.compute_degrees(g)
    #print(degrees)

    # plots?






