import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque
import matplotlib.lines as mlin
from matplotlib.offsetbox import AnchoredText

def plot_degree(g, te_thresh, edge_type, degree_dir, degree_diff_dir):
    '''
    includes: 
    total degree 
    in degree 
    out degree 
    degree difference (out - in) 
    TE weighted degree difference
    '''
    # threshold value text for plots
    #textstr = str('TE-threshold = '+str(te_thresh))
    textstr = str(f"TE-threshold = {te_thresh}")
    
    #number of unique nodes
    num_nodes = g.number_of_nodes()

    ##### TOTAL DEGREE #####
    x = nx.degree(g)
    degree_values = [v for k, v in x]
    nodes = [k for k, v in x]
    plt.hist(degree_values)
    #plt.title(str(edge_type + ' ' + 'degree'))
    plt.title(f"{edge_type} degree")
    plt.ylabel('number of nodes'); plt.xlabel('degree')
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    plt.xlim([-50,50]); plt.ylim([0,80])
    #plt.savefig(str(degree_dir + edge_type + '-degree.jpg'))
    plt.savefig(f"{degree_dir}{edge_type}-degree.jpg")
    plt.clf()

    ##### IN DEGREE #####
    x_in = g.in_degree()
    in_degree_values = [v for k, v in x_in]
    in_nodes = [k for k, v in x_in]
    plt.hist(in_degree_values)
    #plt.title(str(edge_type+' '+'in-degree'))
    plt.title(f"{edge_type} degree")
    plt.ylabel('number of nodes'); plt.xlabel('degree')
    plt.xlim([-50,50]); plt.ylim([0,80])
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    #plt.savefig(str(degree_dir + edge_type + '-in-degree.jpg'))
    plt.savefig(f"{degree_dir}{edge_type}-in-degree.jpg")
    plt.clf()

    ##### OUT DEGREE #####
    x_out = g.out_degree()
    out_degree_values = [v for k, v in x_out]
    out_nodes = [k for k, v in x_out]
    # Pull out values of highest nodes
    #if(edge_type=='UF_TM'):
    #    y = sorted(x_out, key=lambda x: x[1], reverse=True)
    #    print("Out degree rank of UF-TM network")
    #    print(y)
    plt.hist(out_degree_values)
    #plt.title(str(edge_type + ' ' + 'out-degree'))
    plt.title(f"{edge_type} out-degree")
    plt.ylabel('number of nodes'); plt.xlabel('degree')
    plt.xlim([-50,50]); plt.ylim([0,80])
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    #plt.savefig(str(degree_dir + edge_type + '-out-degree.jpg'))
    plt.savefig(f"{degree_dir}{edge_type}-out-degree.jpg")
    plt.clf()

    ##### DEGREE DIFFERENCE #####
    amp = []
    for p in np.arange(num_nodes):
        if (p in (out_nodes)) and (p in (in_nodes)): 
            index = out_nodes.index(p)
            inval = in_degree_values[index]
            outval = out_degree_values[index] 
            amp.append(out_degree_values[index]-in_degree_values[index])
        else:
            amp.append(0)    
    plt.hist(amp)
    #plt.title(str(edge_type + ' ' + 'degree difference (out - in)'))
    plt.title(f"{edge_type} degree difference (out-in)")
    if(max(amp)>abs(min(amp))):
        amp_max = max(amp) +1
    elif(max(amp)<abs(min(amp))):
        amp_max = abs(min(amp)) +1
    plt.ylabel('number of nodes'); plt.xlabel('outreach')
    plt.xlim([-50,50]); plt.ylim([0,80])
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    #plt.savefig(str(degree_diff_dir + edge_type+'-outreach'))
    plt.savefig(f"{degree_diff_dir}{edge_type}-outreach")
    plt.clf()     


def plot_betweenness_centrality(g, te_thresh, edge_type, centrality_dir):
    '''
    includes plots for:
    node betweenness centrality
    edge betweenness centrality
    '''
    #textstr = str('TE-threshold = '+str(te_thresh))
    textstr = f"TE-threshold = {te_thresh}"
    
    #BETWEENNESS CENTRALITY FOR NODES
    BC_nodes = nx.betweenness_centrality(g, normalized = True)
    cent_node_values = BC_nodes.values()
    cent_nodes = BC_nodes.keys()
    plt.hist(cent_node_values,bins=20)
    #plt.title(str(edge_type + ' ' + 'node betweenness centrality'))
    plt.title(f"{edge_type} node betweenness centrality ")
    plt.ylabel('number of nodes'); plt.xlabel('centrality')
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    #plt.savefig(str(centrality_dir + edge_type + '-nodes-centrality.jpg'))
    plt.savefig(f"{centrality_dir}{edge_type}-nodes-centrality.jpg")
    plt.clf()

    # BETWEENNESS CENTRALITY FOR EDGES
    BC_edges = nx.edge_betweenness_centrality(g, normalized = True)
    cent_edge_values = BC_edges.values()
    cent_edges = BC_edges.keys()
    cent_edges_list = []
    for c in cent_edges:
        cent_edges_list.append(str(c))
    plt.hist(cent_edge_values, bins=20)
    #plt.title(str(edge_type + ' ' + 'edge betweenness centrality'))
    plt.title(f"{edge_type} edge betweenness centrality")
    plt.ylabel('number of edges'); plt.xlabel('centrality')
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    #plt.savefig(str(centrality_dir + edge_type +'-edges-centrality.jpg'))
    plt.savefig(f"{centrality_dir}{edge_type}-edges-centrality.jpg")
    plt.clf()

def plot_longest_path(graphs, longest_path_dir):
    paths = []
    for edge_type, graph in graphs.items():
        path_length = nx.dag_longest_path(graph)
        paths.append((edge_type,path_length))
    plt.bar(path[0],path[1])
    plt.savefig('longest_paths.jpg')

def plot_graphs(graphs, graphs_dir, actors, edge_type, te_val, plot_type=None):
    for edge_type, graph in graphs.items():
        nx.relabel_nodes(graph,actors,copy=False)
        if(plot_type):
            #pos=nx.spring_layout(graph)
            #nx.draw(graph,pos)
            nx.draw_circular(graph,with_labels=True)
        else:
            nx.draw(graph)
        #plt.savefig(str(graphs_dir + str(edge_type) + 'te-' + str(te_val) + '-graph.jpg'))
        plt.savefig(f"{graphs_dir}{edge_type}-te-{te_val}-graph.jpg")
        plt.clf()


def plot_path_lengths(lengths, edge_type, te_thresh, paths_dir):
    '''
    lengths is a dictionary {root_node:length}
    '''
    root_nodes = []
    path_length = []
    
    for root_node, length in sorted(lengths.items()):
        root_nodes.append(root_node)
        path_length.append(length)
    root_nodes = [str(x) for x in root_nodes]

    plt.bar(x=root_nodes, height=path_length)
    plt.xlabel("root nodes"); plt.ylabel("length")
    plt.ylim([0,10]) 
    plt.title(str(edge_type + " path lengths"  + " te_thresh=" + str(te_thresh))) 
    #textstr= str("te_thresh=" + str(te_thresh)) 
    textstr= f"te_tresh={te_thresh}"
    plt.rcParams.update({'font.size': 26})
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    #plt.savefig(str(paths_dir + edge_type + "_te-" + str(te_thresh) + "_paths.jpg"))
    plt.savefig(f"{paths_dir}{edge_type}-te-{te_thresh}-paths.jpg")
    plt.clf() 


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


def influential_node_ranking(g, pulltop=0, node_names=False):
    '''
    ranking nodes in a network based on betweenness-centrality
    '''
    BC_nodes = nx.betweenness_centrality(g, normalized = True)
    sorted_nodes = sorted(BC_nodes.items(), key=lambda x:x[1], reverse=True)
    if pulltop != 0:
        sorted_nodes = sorted_nodes[:pulltop]
    if node_names == True:
        for n, i in enumerate(sorted_nodes):
            sorted_nodes[n] = i[0]
    return sorted_nodes


def plot_htrees(graphs, tree_dir, edge_type,te_thresh, actors, visited_lim, depth_lim, orig_nodes, path=None):
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
        plt.savefig(f"{tree_dir}{edge_type}-te-{te_thresh}-root-{root_orig}-tree.jpg")
        plt.clf()



def te_rollout(in_roots, in_edges_df, max_visits=6): 
    lengths = {}
    te_levels_df = {}
    all_root_dfs = {}
    for in_root in in_roots:
        visited = {}
        root_df = pd.DataFrame()
        for node in range(150):
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
                lengths.update({in_root:this_level})
                break
            this_level += 1
            edges_from_this_level = in_edges_df[in_edges_df['Source'].isin(this_level_nodes)]

            visited_cap = set([k for k, v in visited.items() if v > max_visits])
            e = edges_from_this_level[~edges_from_this_level['Target'].isin(visited_cap)]
            root_df = root_df.append(e, ignore_index=True)
            this_level_nodes = set(edges_from_this_level['Target'].to_list()).difference(visited_cap)

        all_root_dfs.update({in_root:root_df})
    
    return lengths, all_root_dfs


def te_rollout_addnodes(in_roots, in_edges_df, max_visits, actors):
    # number of added nodes
    #TODO have appended number in unexpanded node names represent 
    # number of times it shows up, instead of random value
    n=0
    lengths = {}
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
                lengths.update({in_root:this_level})
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
                    #actors[new_node] = str(actor_name + '_' + str(n)) 
                    actors[new_node] = f"{actor_name}_{n}"
                    nodepos = ((e['Source']==from_node) & (e['Target']==to_node))
                    e.loc[nodepos, ['Target']]=new_node
            visited_cap = set([k for k, v in visited.items() if v > max_visits])
            e = e[~e['Target'].isin(visited_cap)]
            root_df = root_df.append(e, ignore_index=True)
            this_level_nodes = set(e['Target'].to_list()).difference(visited_cap)
        all_root_dfs.update({in_root:root_df})
    
    return lengths, all_root_dfs, actors



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
