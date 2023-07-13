import networkx as nx
from . import strongest_path_greedy, strongest_path_summed
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque


def htrees(graphs, edge_type, te_thresh, visited_lim, depth_lim, orig_nodes, path=None):
    '''
    horizontal trees/hierarchical directed graph propogation
    input:
    ...
    path: strongest pathway selection method: None, greedy, or summed (total edge weight)
    '''
    rnodes = []
    xtrees = []
    xpathways = []
    xcolormap_n = []
    xcolormap_e = []
    xpos = []
    xstrengths = []

    for root, graph in graphs.items():
        if not graph.has_node(root):
            continue
        tree_edges = list(graph.edges)
        tree = bfs_tree_AB(G=graph, source=root, visited_lim=visited_lim, depth_lim = depth_lim, edges = tree_edges)
        root_orig = root
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
            pathway, strength = strongest_path_greedy.strongest_path_greedy(tree,graph,root)
        elif path == 'summed':
            pathway, strength = strongest_path_summed.strongest_path_summed(tree,graph,root)
        colormap_edges = []
        for edge in tree.edges:
            if(edge in pathway):
                colormap_edges.append('red')
            else:
                colormap_edges.append('black')
        pos = graphviz_layout(tree, prog='dot', args="-Grankdir=LR")
        rnodes.append(root)
        xtrees.append(tree)
        xpathways.append(pathway)
        xcolormap_n.append(colormap_nodes)
        xcolormap_e.append(colormap_edges)
        xstrengths.append(strength)
        xpos.append(pos)

    return rnodes, xtrees, xpathways, xstrengths, xcolormap_n, xcolormap_e, xpos




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





