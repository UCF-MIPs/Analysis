import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque
import matplotlib.lines as mlin
from matplotlib.offsetbox import AnchoredText


def strongest_path_greedy(tree,graph,root):
    '''
    returns a list of edges to color
    path selected by next strongest edge
    '''
    pathway = []
    # Get layers from tree
    edges_from_source = tree.out_edges(root)
    out_edges = {}
    for i in edges_from_source:
        n = list(graph.get_edge_data(*i).values())
        out_edges[i]=n
    max_val = max(out_edges.values())
    selection = (k for k, v in out_edges.items() if v == max_val)
    elem = [*selection]
    f,t = elem[0]
    pathway.append((f,t))
    in_node = t
    #traverse layers
    while(True):
        edges_from_in = tree.out_edges(in_node)
        if not edges_from_in:
            break
        out_edges= {}
        for i in edges_from_in:
            n = list(graph.get_edge_data(*i).values())
            out_edges[i]=n
        max_val = max(out_edges.values())
        selection = (k for k, v in out_edges.items() if v== max_val)
        elem = [*selection]
        f,t = elem[0]
        pathway.append((f,t))
        in_node = t
    return pathway


def strongest_path_summed(tree, graph, root):
    '''
    returns a list of edges to color
    path selected by summed TE over all possible paths
    '''

    pathways = []
    roots = []
    leaves = []
    for node in tree.nodes:
        if tree.in_degree(node) == 0:
            roots.append(node)
        elif tree.out_degree(node) == 0 :
            leaves.append(node)

    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_edge_paths(tree, root, leaf) :
                pathways.append(path)

    total = 0
    total_temp=0
    for path in pathways:
        for edge in path:
            total_temp += float(list(graph.get_edge_data(*edge).values())[0])
        if total_temp > total:
            total = total_temp
            strongest_pathway = path
        total_temp=0 
    return strongest_pathway


def auto_threshold(df, col, max_nodes):
    '''
    Drops columns not of interest
    Finds weight threshold for a network that results in a given number of nodes
    '''
    df = df[['Source', 'Target', col]]
    for thresh in np.round(np.linspace(0. ,1. ,41, endpoint=True), 3):
        df_filtered = df.loc[(df[col] > thresh)]
        num_nodes = df_filtered['Source'].nunique()
        print(num_nodes)
        if(num_nodes < max_nodes):
            break
    return thresh, df_filtered


def influential_node_ranking(g, pulltop=0, node_names=False):
    '''
    ranking nodes in a network based on betweenness-centrality
    node_names: bool to indicate whether or not to return just node names:
        True - just return top node names
        False - return node names and betweenness centrality as tuple
    '''
    BC_nodes = nx.betweenness_centrality(g, normalized = True)
    sorted_nodes = sorted(BC_nodes.items(), key=lambda x:x[1], reverse=True)
    if pulltop != 0:
        sorted_nodes = sorted_nodes[:pulltop]
    if node_names == True:
        for n, i in enumerate(sorted_nodes):
            sorted_nodes[n] = i[0]
    return sorted_nodes




def htrees(graphs, edge_type, te_thresh, actors, visited_lim, depth_lim, orig_nodes, path=None):
    '''
    horizontal trees/hierarchical directed graph propogation
    input: 
    ...
    path: strongest pathway selection method: None, greedy, or summed (total edge weight)
    '''
    for root, graph in graphs.items():
        if not graph.has_node(root):
            return
        tree_edges = list(graph.edges)
        tree = bfs_tree_AB(G=graph, source=root, visited_lim=visited_lim, depth_lim = depth_lim, edges = tree_edges)
        nx.relabel_nodes(tree,actors,copy=False)
        nx.relabel_nodes(graph,actors,copy=False)
        if path == None:
            pass
        elif path == 'greedy':
            pathway = strongest_path_greedy(tree,graph,root)
        elif path == 'summed':
            pathway = strongest_path_summed(tree,graph,root)
        return tree, pathway


def plot_htrees(graphs, tree_dir, edge_type, te_thresh, actors, visited_lim, depth_lim, orig_nodes, path=None):
    '''
    horizontal trees/hierarchical directed graph propogation
    input: 
    ...
    path: strongest pathway selection method: None, greedy, or summed (total edge weight)
    '''
    for root, graph in graphs.items():
        if not graph.has_node(root):
            return
        tree_edges = list(graph.edges)
        tree = bfs_tree_AB(G=graph, source=root, visited_lim=visited_lim, depth_lim = depth_lim, edges = tree_edges)
        nx.relabel_nodes(tree,actors,copy=False)
        nx.relabel_nodes(graph,actors,copy=False)
        root_orig = root
        root = actors[root]
        colormap_nodes = []
        for node in tree:
            if(node in orig_nodes):
                if graph.out_degree(node)==0:    
                    colormap_nodes.append('green')
                else: 
                    colormap_nodes.append('#1f78b4')
            elif(node not in orig_nodes):
                    colormap_nodes.append('yellow')
        if path == None:
            pass
        elif path == 'greedy':
            pathway = strongest_path_greedy(tree,graph,root)
        elif path == 'summed':
            pathway = strongest_path_summed(tree,graph,root)
        print(pathway)
        colormap_edges = []
        for edge in tree.edges:
            if(edge in pathway):
                colormap_edges.append('red')
            else:
                colormap_edges.append('black')
        pos = graphviz_layout(tree, prog='dot', args="-Grankdir=LR")
        nx.draw(tree, pos, node_color=colormap_nodes, edge_color=colormap_edges, \
                with_labels=True, width=3, font_size=24, node_size=450)
        
        # output: tree, path



def te_rollout_addnodes(in_roots, in_edges_df, max_visits, actors):
    # number of added nodes
    #TODO have appended number in unexpanded node names represent 
    # number of times it shows up, instead of random value
    n=0
    te_levels_df = {}
    all_root_dfs = {}
    for in_root in in_roots:
        visited = {}
        root_df = pd.DataFrame()
        for node in range(10000):
            visited.update({node:0})
        this_level_nodes = in_root
        te_values = []
        this_level = 0
        while True:
            if(this_level==0):
                this_level_nodes = [this_level_nodes]
            last_visited = visited.copy()
            for node in this_level_nodes:
                visited[node] += 1
            if(last_visited == visited):
                break
            this_level += 1
            e = in_edges_df[in_edges_df['Source'].isin(this_level_nodes)]
            # Replace edge to visited node with new edge to new terminal node with same ID
            for index, edge in e.iterrows():
                from_node = edge['Source']
                to_node = edge['Target']
                if visited[to_node]>0:
                    #add new edge to new node with same outgoing actor ID
                    new_node = 120 + n
                    n+=1
                    actor_name = actors[to_node]
                    actors[new_node] = f"{actor_name}_{n}"
                    nodepos = ((e['Source']==from_node) & (e['Target']==to_node))
                    e.loc[nodepos, ['Target']]=new_node
            visited_cap = set([k for k, v in visited.items() if v > max_visits])
            e = e[~e['Target'].isin(visited_cap)]
            root_df = root_df.append(e, ignore_index=True)
            this_level_nodes = set(e['Target'].to_list()).difference(visited_cap)
        all_root_dfs.update({in_root:root_df})
    
    return all_root_dfs, actors



# OVERRIDES

def bfs_tree_AB(G, source, visited_lim, depth_lim, reverse=False, depth_limit=None, sort_neighbors=None, edges=None):
    T = nx.DiGraph()
    T.add_node(source)
    edges_gen = bfs_edges_AB(
        G,
        source,
        visited_lim=visited_lim,
        depth_lim=depth_lim,
        reverse=reverse,
        depth_limit=depth_limit,
        sort_neighbors=sort_neighbors,
    )
    T.add_edges_from(edges)
    return T


def generic_bfs_edges_AB(G, source, visited_lim, depth_lim, neighbors=None, depth_limit=None, sort_neighbors=None):
    if callable(sort_neighbors):
        _neighbors = neighbors
        neighbors = lambda node: iter(sort_neighbors(_neighbors(node)))
    visited = {}
    for i in range(1000):
        visited[i]=0

    if depth_limit is None:
        #depth_limit = len(G)
        depth_limit = depth_lim
    queue = deque([(source, depth_limit, neighbors(source))])
    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if visited[child] < visited_lim:
                yield parent, child
                visited[child] += 1
                if depth_now > 1:
                    queue.append((child, depth_now - 1, neighbors(child)))
        
        except StopIteration:
            queue.popleft()


def bfs_edges_AB(G, source, visited_lim, depth_lim, reverse=False, depth_limit=None, sort_neighbors=None):
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    yield from generic_bfs_edges_AB(G, source, visited_lim, depth_lim, successors, depth_limit, sort_neighbors)
