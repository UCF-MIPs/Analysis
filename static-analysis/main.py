import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .src import auto_threshold
from .src import te_rollout
from .src import htrees
from .src import plot_htrees
from .src import influential_node_ranking
from .src import influential_edge_ranking
from .src import generate_edge_types
from .src import generate_trees
      
# Use in interface
pathway_selection="greedy" # options: summed, greedy, or None
edge_type = 'TM_TM'
te_thresh = 0.1
#node_rank = 'bc'
node_rank='manual' # options: bc, outdegree, manual. If manual, populate manual_root with name
manual_root = ['BombshellDAILY'] # must be a list, can include any number of desired root nodes
dataset = 'ukr_v3' # options: skrip_v4, skrip_v7, ukr_v3

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

# Results dir
if node_rank == 'manual':
    dir_name = f'results/{dataset}_{node_rank}_{manual_root}_{pathway_selection}'
else:
    dir_name = f'results/{dataset}_{node_rank}_{pathway_selection}'
path = Path(dir_name)
path.mkdir(parents=True, exist_ok=True)

for chunk in pd.read_csv(te_df_path, chunksize=1000, usecols=['Source', 'Target', edge_type]):
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

actor_df = pd.read_csv(act_df_path)
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
nx.relabel_nodes(g, actors, copy=False)

# Can replace 'influential_node_ranking with a single root node as a list
# for example, root_nodes = ['Ian56789']

# note: if using generate_trees, node_names must be set to True
root_nodes = influential_node_ranking.influential_node_ranking_outdegree(g, pulltop=5, node_names=True)


if node_rank == 'outdegree':
    root_nodes = influential_node_ranking.influential_node_ranking_outdegree(g, pulltop=5, node_names=True)
if node_rank == 'bc':
    root_nodes = influential_node_ranking.influential_node_ranking_bc(g, pulltop=5, node_names=True)
if node_rank == 'manual':
    root_nodes = manual_root

print(f'{te_thresh} {edge_type}')
print(root_nodes)

generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name=None)

#rnodes, xtrees, xpathways, xstrengths, xcolormap_nodes, xcolormap_edges, xpos = generate_trees.generate_tree_data(g, edge_type, te_thresh, pathway_selection, root_nodes)

