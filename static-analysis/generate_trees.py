import networkx as nx
import pandas as pd
from src import auto_threshold
from src import te_rollout
from src import htrees
from src import plot_htrees
from src import influential_node_ranking
from src import influential_edge_ranking
from src import generate_edge_types


def generate_trees(g, edge_type, te_thresh, pathway_selection):

    vis_lim = 2
    dep_lim = 7
    orig_nodes = g.nodes

    root_nodes = [x for x in g.nodes() if g.out_degree(x) > 0]
    g_df = nx.to_pandas_edgelist(g, source='Source', target='Target')
    
    root_graphs = {}
    for in_root in root_nodes:
        root_df = te_rollout.te_rollout(in_root, g_df, vis_lim)
        g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        root_graphs.update({in_root:g})

    # Generate tree information in for of lists (1 entry per root node)
    rnodes, xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos = htrees.htrees(root_graphs, edge_type, te_thresh, vis_lim, dep_lim, orig_nodes, path=pathway_type) 
    ts_figs = plot_htrees.plot_htrees(xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos, te_thresh, edge_type)

    # Save resulting tree plots
    print(ts_figs)
    for ax, root in zip(ts_figs, rnodes):
        ax.figure.savefig(f'{edge_type}_te_thresh{te_thresh}_root{root}.png')


def generate_tree_data(g, edge_type, te_thresh, pathway_type):

    vis_lim = 2
    dep_lim = 7
    orig_nodes = g.nodes

    root_nodes = [x for x in g.nodes() if g.out_degree(x) > 0]
    g_df = nx.to_pandas_edgelist(g, source='Source', target='Target')
    
    root_graphs = {}
    for in_root in root_nodes:
        root_df = te_rollout.te_rollout(in_root, g_df, vis_lim)
        g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        root_graphs.update({in_root:g})

    # Generate tree information in for of lists (1 entry per root node)
    rnodes, xtrees, xpathways, xstrengths, xcolormap_nodes, xcolormap_edges, xpos = htrees.htrees(root_graphs, edge_type, te_thresh, vis_lim, dep_lim, orig_nodes, path=pathway_type) 

    return xtrees, xpathways, xstrengths



'''
# input variables
edge_type = 'UF_UM' # options: ...
pathway_type="greedy" # options: summed, greedy, or None

# Dataframe of TE network
cascade_df = pd.read_csv('data/actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('data/actors_v4.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

#te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 80, return_df=True)
#te_thresh = auto_threshold.auto_threshold(cascade_df, edge_type, 80, return_df=False)

te_thresh = 0.1
graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
nx.relabel_nodes(g, actors, copy=False)

generate_trees(g, edge_type, te_thresh)
'''

