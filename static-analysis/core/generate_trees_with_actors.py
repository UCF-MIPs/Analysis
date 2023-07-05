import networkx as nx
import pandas as pd
from src import auto_threshold
from src import te_rollout_addnodes
from src import te_rollout_addnodes_with_actors
from src import htrees
from src import htrees_with_actors
from src import plot_htrees
from src import influential_node_ranking
from src import influential_edge_ranking
from src import generate_edge_types


def generate_trees_with_actors(g, edge_type, te_thresh):


    pathway_type="greedy" # options: summed, greedy, or None
    vis_lim = 3
    dep_lim = 5
    #actors  = #? 

    #root_nodes = list(g.nodes)
    root_nodes = [x for x in g.nodes() if g.out_degree(x) > 0]
    print(root_nodes)

    g_df = nx.to_pandas_edgelist(g, source='Source', target='Target')

    all_root_dfs = te_rollout_addnodes_with_actors.te_rollout_addnodes_with_actors(in_roots = root_nodes, in_edges_df = g_df, max_visits=vis_lim)

    # Graph/tree plotting of paths from root
    root_graphs = {}
    for roots, root_df in all_root_dfs.items():
        g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        root_graphs.update({roots:g})

    # Generate tree information in for of lists (1 entry per root node)
    rnodes, xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos = htrees_with_actors.htrees_with_actors(root_graphs, edge_type, te_thresh, vis_lim, dep_lim, orig_nodes, path=pathway_type) 

    ts_figs = plot_htrees.plot_htrees(xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos, te_thresh, edge_type)

    # Save resulting tree plots
    print(ts_figs)
    for ax, root in zip(ts_figs, rnodes):
        print(root)
        # check to see if root node is mapped somewhere
        ax.figure.savefig(f'{edge_type}_te_thresh{te_thresh}_root{root}.png')

 

# input variables
edge_type = 'UM_TM' # options: ...

# Dataframe of TE network
cascade_df = pd.read_csv('actor_te_edges_2018_03_01_2018_05_01.csv')
cascade_df = cascade_df.loc[(cascade_df['Source'] > 1.) & (cascade_df['Target'] > 1.) & (cascade_df['Source']<101.) & (cascade_df['Target']<101.)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('actors.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))
actors_orig = actors
orig_nodes = list(actors_orig.values())

te_thresh, graph_df = auto_threshold.auto_threshold(cascade_df, edge_type, 80)

g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())

nx.relabel_nodes(g, actors, copy=False)

generate_trees_with_actors(g, edge_type, te_thresh)


