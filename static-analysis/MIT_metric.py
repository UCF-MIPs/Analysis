import networkx as nx
import pandas as pd
from src import generate_edge_types
import matplotlib.pyplot as plt

# input variables
pathway_selection = "greedy"

# Dataframe of TE network
cascade_df = pd.read_csv('data/actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('data/actors_v4.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

#te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 80, return_df=True)
#te_thresh = auto_threshold.auto_threshold(cascade_df, edge_type, 80, return_df=False)

# Create df to store results
infl_df = pd.DataFrame()

edge_types = generate_edge_types.generate_edge_types()
num_edges_over_thresh = {}

for edge_type in edge_types: 
    num_edges = []
    for te_thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
        g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        nx.relabel_nodes(g, actors, copy=False)
        num_edges.append(len(g.edges))
    num_edges_over_thresh[edge_type] = sum(num_edges)

print(num_edges_over_thresh)



