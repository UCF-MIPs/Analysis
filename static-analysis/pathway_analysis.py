from src import generate_trees
import networkx as nx
import pandas as pd
from src import generate_edge_types
import matplotlib.pyplot as plt

# input variables
pathway_selection = "greedy"

# Dataframe of TE network
cascade_df = pd.read_csv('data/Skripal/actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('data/Skripal/actors_v4.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

#te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 80, return_df=True)
#te_thresh = auto_threshold.auto_threshold(cascade_df, edge_type, 80, return_df=False)

# Create df to store results
infl_df = pd.DataFrame()

edge_types = generate_edge_types.generate_edge_types() 
av_pathway_lengths = {}
longest_pathway_lengths = {}
av_pathway_weights = {}
strongest_pathway_weights = {}


for edge_type in edge_types:
    te_thresh = 0.1
    graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
    g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
    nx.relabel_nodes(g, actors, copy=False)
    xtrees, xpathways, xstrengths = generate_trees.generate_tree_data(g, edge_type, te_thresh, pathway_selection)
    pathway_lens = []
    #for tree in xtrees:
    #    print(tree)
    longest_pathway = 0
    for pathway in xpathways:
        #print(len(pathway))
        pathway_lens.append(len(pathway))
        if len(pathway) > longest_pathway:
            longest_pathway = len(pathway)
    path_av = sum(pathway_lens)/len(pathway_lens)
    av_pathway_lengths[edge_type] = path_av
    longest_pathway_lengths[edge_type] = longest_pathway

names = list(av_pathway_lengths.keys())
values = list(av_pathway_lengths.values())

plt.ion()
plt.bar(range(len(av_pathway_lengths)), values)
plt.xticks(range(len(av_pathway_lengths)), names, rotation='vertical')
plt.ylabel('path length')
plt.title('Average pathway length')
plt.savefig('plots/ave_path_lengths.png')
plt.clf()


names = list(longest_pathway_lengths.keys())
values = list(longest_pathway_lengths.values())

plt.bar(range(len(longest_pathway_lengths)), values)
plt.xticks(range(len(longest_pathway_lengths)), names, rotation='vertical')
plt.ylabel('path length')
plt.title('Longest pathway length')
plt.savefig('plots/longest_path_lengths.png')
plt.clf()
# weights

pathway_selection = "summed"


for edge_type in edge_types:
    te_thresh = 0.1
    graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
    g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
    nx.relabel_nodes(g, actors, copy=False)
    xtrees, xpathways, xstrengths = generate_trees.generate_tree_data(g, edge_type, te_thresh, pathway_selection)
    pathway_weights = []
    #for tree in xtrees:
    #    print(tree)
    strongest_weight = 0
    for pathway, strength in zip(xpathways, xstrengths):
        pathway_weights.append(strength)
        if strength > strongest_weight:
            print(strength)
            strongest_weight = strength
    path_av = sum(pathway_weights)/len(pathway_weights)
    av_pathway_weights[edge_type] = path_av
    strongest_pathway_weights[edge_type] = strongest_weight


names = list(av_pathway_weights.keys())
values = list(av_pathway_weights.values())

plt.bar(range(len(av_pathway_weights)), values)
plt.xticks(range(len(av_pathway_weights)), names, rotation='vertical')
plt.ylabel('path weight')
plt.title('Average pathway weights')
plt.savefig('plots/ave_path_weights.png')
plt.clf()


names = list(strongest_pathway_weights.keys())
values = list(strongest_pathway_weights.values())

plt.bar(range(len(strongest_pathway_weights)), values)
plt.xticks(range(len(strongest_pathway_weights)), names, rotation='vertical')
plt.ylabel('path weight')
plt.title('Strongest pathway weights')
plt.savefig('plots/strongest_path_weights.png')
plt.clf()

path_metrics_df = pd.DataFrame.from_dict([av_pathway_lengths, longest_pathway_lengths, av_pathway_weights, strongest_pathway_weights])

print(path_metrics_df)

path_metrics_transposed = path_metrics_df.T

print(path_metrics_transposed)

path_metrics_transposed.columns = ['av_path_length', 'longest_path_length', 'av_path_weight', 'strongest_path_weight']

path_metrics_transposed.to_csv('path_metrics.csv')


