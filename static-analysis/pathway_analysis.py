from src import generate_trees
import networkx as nx
import pandas as pd
from src import generate_edge_types
import matplotlib.pyplot as plt

# input variables
pathway_selection = "greedy"
edge_types = generate_edge_types.generate_edge_types() 

av_pathway_lengths = {}
longest_pathway_lengths = {}
av_pathway_weights = {}
strongest_pathway_weights = {}

infl_df = pd.DataFrame(edge_types, columns = ['infl_type'])

print(infl_df)

te_threshes = [0.05, 0.075, 0.1]

for te_thresh in te_threshes:
    for edge_type in edge_types:    
        #cascade_df = pd.read_csv('data/Ukraine/Actor_TE_Edges_Ukraine_v1.csv', usecols=['Source', 'Target', edge_type])
        iter_csv = pd.read_csv('data/Ukraine/Actor_TE_Edges_Ukraine_v1.csv', iterator=True, chunksize=1000, usecols=['Source', 'Target', edge_type])
        graph_df = pd.concat([chunk[chunk[edge_type] > te_thresh] for chunk in iter_csv])
        actor_df = pd.read_csv('data/Ukraine/actors_Ukraine_v1.csv')
        actors = dict(zip(actor_df.actor_id, actor_df.actor_label))
        g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        nx.relabel_nodes(g, actors, copy=False)
        rnodes, xtrees, xpathways, xstrengths, xcolormap_nodes, xcolormap_edges, xpos = generate_trees.generate_tree_data(g, edge_type, te_thresh, pathway_selection, root_nodes=None)
        pathway_lens = []
        longest_pathway = 0
        for pathway in xpathways:
            #print(len(pathway))
            pathway_lens.append(len(pathway))
            if len(pathway) > longest_pathway:
                longest_pathway = len(pathway)
        if(len(pathway_lens) != 0):
            path_av = sum(pathway_lens)/len(pathway_lens)
        else:
            path_av = 0
        av_pathway_lengths[edge_type] = path_av
        longest_pathway_lengths[edge_type] = longest_pathway
        pathway_weights = []
        strongest_weight = 0
        for pathway, strength in zip(xpathways, xstrengths):
            pathway_weights.append(strength)
            if strength > strongest_weight:
                strongest_weight = strength
        if(len(pathway_lens) !=0):
            path_av = sum(pathway_weights)/len(pathway_weights)
        else:
            path_av = 0
        av_pathway_weights[edge_type] = path_av
        strongest_pathway_weights[edge_type] = strongest_weight 
        print(f'{edge_type}_{te_thresh}')
    infl_df[f'strongest_path_weight_{te_thresh}'] = infl_df['infl_type'].apply(lambda x: strongest_pathway_weights.get(x)).fillna('')
    infl_df[f'av_path_weight_{te_thresh}'] = infl_df['infl_type'].apply(lambda x: av_pathway_weights.get(x)).fillna('')
    infl_df[f'longest_path_len_{te_thresh}'] = infl_df['infl_type'].apply(lambda x: longest_pathway_lengths.get(x)).fillna('')
    infl_df[f'av_path_len_{te_thresh}'] =  infl_df['infl_type'].apply(lambda x: av_pathway_lengths.get(x)).fillna('')
    print(infl_df)

'''
# weights

pathway_selection = "summed"

for te_thresh in te_threshes:
    for edge_type in edge_types:
        #cascade_df = pd.read_csv('data/Ukraine/Actor_TE_Edges_Ukraine_v1.csv', usecols=['Source', 'Target', edge_type])
        iter_csv = pd.read_csv('data/Ukraine/Actor_TE_Edges_Ukraine_v1.csv', iterator=True, chunksize=1000, use_cols=['Source', 'Target', edge_type])
        cascade_df = pd.concat([chunk[chunk[edge_type] > te_thresh] for chunk in iter_csv])
        actor_df = pd.read_csv('data/Ukraine/actors_Ukraine_v1.csv')
        actors = dict(zip(actor_df.actor_id, actor_df.actor_label))
        #graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
        g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        nx.relabel_nodes(g, actors, copy=False)
        rnodes, xtrees, xpathways, xstrengths, xcolormap_nodes, xcolormap_edges, xpos = generate_trees.generate_tree_data(g, edge_type, te_thresh, pathway_selection, root_nodes=None)
        pathway_weights = []
        strongest_weight = 0
        for pathway, strength in zip(xpathways, xstrengths):
            pathway_weights.append(strength)
            if strength > strongest_weight:
                strongest_weight = strength
        path_av = sum(pathway_weights)/len(pathway_weights)
        av_pathway_weights[edge_type] = path_av
        strongest_pathway_weights[edge_type] = strongest_weight 
        print(f'{edge_type}_{te_thresh}')
    infl_df[f'strongest_path_weight_{te_thresh}'] = infl_df['infl_type'].apply(lambda x: strongest_pathway_weights.get(x)).fillna('')
    infl_df[f'av_path_weight_{te_thresh}'] = infl_df['infl_type'].apply(lambda x: av_pathway_weights.get(x)).fillna('')
    print(infl_df)
'''

cols = ['infl_type', \
        'longest_path_len_0.025', \
        'longest_path_len_0.05', \
        'longest_path_len_0.075',   \
        'longest_path_len_0.1', \
        'av_path_len_0.025', \
        'av_path_len_0.05', \
        'av_path_len_0.075', \
        'av_path_len_0.1', \
        'strongest_path_weight_0.025', \
        'strongest_path_weight_0.05', \
        'strongest_path_weight_0.075', \
        'strongest_path_weight_0.1', \
        'av_path_weight_0.025', \
        'av_path_weight_0.05', \
        'av_path_weight_0.075', \
        'av_path_weight_0.1']

infl_df = infl_df[cols]
infl_df.to_csv('pathway_analysis_Ukraine.csv', index=False)
