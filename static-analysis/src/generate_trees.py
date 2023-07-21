import networkx as nx
import pandas as pd
from . import te_rollout, htrees, plot_htrees


def generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name):

    vis_lim = 2
    dep_lim = 7
    orig_nodes = g.nodes
    if root_nodes is None:
        root_nodes = [x for x in g.nodes() if g.out_degree(x) > 0]
    g_df = nx.to_pandas_edgelist(g, source='Source', target='Target')
    
    root_graphs = {}
    for in_root in root_nodes:
        root_df = te_rollout.te_rollout(in_root, g_df, vis_lim)
        g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
        root_graphs.update({in_root:g})

    # Generate tree information in for of lists (1 entry per root node)
    rnodes, xtrees, xpathways, xstrengths, xcolormap_nodes, xcolormap_edges, xpos = htrees.htrees(root_graphs, edge_type, te_thresh, vis_lim, dep_lim, orig_nodes, path=pathway_selection) 
    ts_figs = plot_htrees.plot_htrees(xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos, te_thresh, edge_type)

    # Save resulting tree plots
    print(ts_figs)
    for ax, root in zip(ts_figs, rnodes):
        if dir_name is not None:
            ax.figure.savefig(f'{dir_name}/{edge_type}_te_thresh{te_thresh}_root{root}.png')
        else: 
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

