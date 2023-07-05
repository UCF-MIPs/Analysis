import networkx as nx
import pandas as pd
from src import auto_threshold
from src import te_rollout_addnodes
from src import htrees
from src import plot_htrees
from src import influential_node_ranking
from src import influential_edge_ranking
from src import generate_edge_types


def generate_trees(g, actors, edge_type, te_thresh):


    pathway_type="greedy" # options: summed, greedy, or None
    vis_lim = 3
    dep_lim = 5
    #actors  = #? 

    #root_nodes = list(g.nodes)
    root_nodes = [x for x in g.nodes() if g.out_degree(x) > 0]
    print(root_nodes)

    g_df = nx.to_pandas_edgelist(g, source='Source', target='Target')

    all_root_dfs, actors = te_rollout_addnodes.te_rollout_addnodes(in_roots = root_nodes, in_edges_df = g_df, max_visits=vis_lim, actors=actors)

    # Graph/tree plotting of paths from root
    root_graphs = {}
    for roots, root_df in all_root_dfs.items():
        g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        root_graphs.update({roots:g})

    # Generate tree information in for of lists (1 entry per root node)
    rnodes, xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos = htrees.htrees(root_graphs, edge_type, te_thresh, actors, vis_lim, dep_lim, orig_nodes, path=pathway_type)

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

print(actors)

generate_trees(g, actors, edge_type, te_thresh)



