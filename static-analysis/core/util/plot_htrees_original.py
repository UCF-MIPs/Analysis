import maplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque
import matplotlib.lines as mlin
from matplotlib.offsetbox import AnchoredText

def plot_htrees_original(graphs, tree_dir, edge_type, te_thresh, actors, visited_lim, depth_lim, orig_nodes, path=None):
    '''
    horizontal trees/hierarchical directed graph propogation
    input:
    ...
    path: strongest pathway selection method: None, greedy, or summed (total edge weight)
    '''
    plt.clf()

    for root, graph in graphs.items():
        if not graph.has_node(root):
            continue
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
        #short
        #plt.figure(3,figsize=(20,20))
        #tall
        plt.figure(3,figsize=(18,50))
        node_type = ['Expanded', 'Terminal', 'Unexpanded']
        te_text = str('TE threshold: ' + str(te_thresh))
        text_box = AnchoredText(te_text, frameon=True, loc='lower left', pad=0.5)
        plt.setp(text_box.patch, facecolor='white', alpha=0.5)
        plt.gca().add_artist(text_box)
        te_text2 = str('Influence type: ' + '\n' + str(edge_type))
        text_box2 = AnchoredText(te_text2, frameon=True, loc='lower center', pad=0.5)
        plt.gca().add_artist(text_box2)

        line1 = mlin.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="#1f75ae")
        line2 = mlin.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="green")
        line3 = mlin.Line2D([], [], color="white", marker='o', markersize=15,  markerfacecolor="yellow")
        plt.legend((line1, line2, line3), ('Expanded', 'Terminal', 'Unexpanded'), numpoints=1, loc='lower right')

        #plt.savefig(str(tree_dir + edge_type + "-te-" + str(te_thresh) + "-root-"+ str(root_orig) + '-tree.jpg'))
        #plt.savefig(f"{tree_dir}{edge_type}-te-{te_thresh}-root-{root_orig}-tree.jpg")
        #plt.clf()


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



