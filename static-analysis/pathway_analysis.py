from src import generate_trees
import networkx as nx
import pandas as pd
from src import generate_edge_types
import matplotlib.pyplot as plt
from pathlib import Path


# input variables
pathway_selection = "greedy"
edge_types = generate_edge_types.generate_edge_types() 
#edge_types.remove('TM_TM')
te_threshes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
node_rank = 'outdegree'
dataset = 'skrip_v7' # options: skrip_v4, skrip_v7, ukr_v3

# Data
skrip_v4_te = 'data/Skripal/v4/indv_network/actor_te_edges_2018_03_01_2018_05_01.csv'
skrip_v4_act = 'data/Skripal/v4/indv_network/actors_v4.csv'
# note for skrip v3, need to filter last 10 actors (platforms) like so
#cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

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

# Results dir
dir_name = f'results/{dataset}_{node_rank}_{pathway_selection}'
path = Path(dir_name)
path.mkdir(parents=True, exist_ok=True)

# Results dicts
av_pathway_lengths = {}
longest_pathway_lengths = {}
av_pathway_weights = {}
strongest_pathway_weights = {}

infl_df = pd.DataFrame(edge_types, columns = ['infl_type'])
print(infl_df)

for te_thresh in te_threshes:
    for edge_type in edge_types:    
        iter_csv = pd.read_csv(te_df_path, iterator=True, chunksize=1000, usecols=['Source', 'Target', edge_type])
        graph_df = pd.concat([chunk[chunk[edge_type] > te_thresh] for chunk in iter_csv])
        actor_df = pd.read_csv(act_df_path)
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

cols = ['infl_type', \
        'longest_path_len_0.05', \
        'longest_path_len_0.1', \
        'longest_path_len_0.15',   \
        'longest_path_len_0.2', \
        'longest_path_len_0.25', \
        'longest_path_len_0.3', \
        'av_path_len_0.05', \
        'av_path_len_0.1', \
        'av_path_len_0.15', \
        'av_path_len_0.2', \
        'av_path_len_0.25', \
        'av_path_len_0.3', \
        'strongest_path_weight_0.05', \
        'strongest_path_weight_0.1', \
        'strongest_path_weight_0.15', \
        'strongest_path_weight_0.2', \
        'strongest_path_weight_0.25', \
        'strongest_path_weight_0.3', \
        'av_path_weight_0.05', \
        'av_path_weight_0.1', \
        'av_path_weight_0.15', \
        'av_path_weight_0.2', \
        'av_path_weight_0.25', \
        'av_path_weight_0.3'
        ]

infl_df = infl_df[cols]
infl_df.to_csv(f'pathway_analysis_{dataset}_noderank_{node_rank}.csv', index=False)
