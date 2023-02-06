import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque

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
    textstr = str('TE-threshold = '+str(te_thresh))
    
    #number of unique nodes
    num_nodes = g.number_of_nodes()

    ##### TOTAL DEGREE #####
    x = nx.degree(g)
    degree_values = [v for k, v in x]
    nodes = [k for k, v in x]
    #te_weight = g[str(edge_type)]
    #print(nodes)
    #te_weights = 
    #degree_values = degree_values * te_weight
    plt.hist(degree_values)
    plt.title(str(edge_type + ' ' + 'degree'))
    plt.ylabel('number of nodes')
    plt.xlabel('degree')
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    plt.xlim([-50,50])
    plt.ylim([0,80])
    plt.savefig(str(degree_dir + edge_type + '-degree.jpg'))
    plt.clf()

    ##### IN DEGREE #####
    x_in = g.in_degree()
    in_degree_values = [v for k, v in x_in]
    in_nodes = [k for k, v in x_in]
    plt.hist(in_degree_values)
    plt.title(str(edge_type+' '+'in-degree'))
    plt.ylabel('number of nodes')
    plt.xlabel('degree')
    plt.xlim([-50,50])
    plt.ylim([0,80])
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    plt.savefig(str(degree_dir + edge_type + '-in-degree.jpg'))
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
    plt.title(str(edge_type + ' ' + 'out-degree'))
    plt.ylabel('number of nodes')
    plt.xlabel('degree')
    plt.xlim([-50,50])
    plt.ylim([0,80])
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    plt.savefig(str(degree_dir + edge_type + '-out-degree.jpg'))
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
    plt.title(str(edge_type + ' ' + 'degree difference (out - in)'))
    if(max(amp)>abs(min(amp))):
        amp_max = max(amp) +1
    elif(max(amp)<abs(min(amp))):
        amp_max = abs(min(amp)) +1
    plt.ylabel('number of nodes')
    plt.xlabel('outreach')
    plt.xlim([-50,50])
    plt.ylim([0,80])
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    plt.savefig(str(degree_diff_dir + edge_type+'-outreach'))
    plt.clf()     

    #TODO
    # Weighted degree difference



def plot_betweenness_centrality(g, te_thresh, edge_type, centrality_dir):
    '''
    includes plots for:
    node betweenness centrality
    edge betweenness centrality
    '''
    # threshold value text for plots
    textstr = str('TE-threshold = '+str(te_thresh))

    # BETWEENNESS CENTRALITY FOR NODES
    BC_nodes = nx.betweenness_centrality(g, normalized = True)

    #print some values
    #if(edge_type=='UF_TM'):
    #    y = dict(sorted(BC_nodes.items(), key=lambda item: item[1]))
    #    y = list(reversed(y))
    #    print("Betweenness centrality rank of UF-TM TE network")
    #    print(y)

    cent_node_values = BC_nodes.values()
    cent_nodes = BC_nodes.keys()
    plt.hist(cent_node_values,bins=20)
    plt.title(str(edge_type + ' ' + 'node betweenness centrality'))
    plt.ylabel('number of nodes')
    plt.xlabel('centrality')
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    plt.savefig(str(centrality_dir + edge_type + '-nodes-centrality.jpg'))
    plt.clf()

    # BETWEENNESS CENTRALITY FOR EDGES
    BC_edges = nx.edge_betweenness_centrality(g, normalized = True)
    cent_edge_values = BC_edges.values()
    cent_edges = BC_edges.keys()
    cent_edges_list = []
    for c in cent_edges:
        cent_edges_list.append(str(c))
    plt.hist(cent_edge_values, bins=20)
    plt.title(str(edge_type + ' ' + 'edge betweenness centrality'))
    plt.ylabel('number of edges')
    plt.xlabel('centrality')
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    plt.savefig(str(centrality_dir + edge_type +'-edges-centrality.jpg'))
    plt.clf()

def plot_longest_path(graphs, longest_path_dir):
    paths = []
    for edge_type, graph in graphs.items():
        path_length = nx.dag_longest_path(graph)
        paths.append((edge_type,path_length))
    plt.bar(path[0],path[1])
    plt.savefig('longest_paths.jpg')

def plot_graphs(graphs, graphs_dir, plot_type=None):
    for edge_type, graph in graphs.items():
        if(plot_type):
            #pos=nx.spring_layout(graph)
            #nx.draw(graph,pos)
            nx.draw_circular(graph,with_labels=True)
        else:
            nx.draw(graph)
        plt.savefig(str(graphs_dir + edge_type + '-graph.jpg'))
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
    plt.xlabel("root nodes")
    plt.ylabel("length")
    plt.ylim([0,10])
    plt.title(str(edge_type + " path lengths"  + " te_thresh=" + str(te_thresh))) 
    textstr= str("te_thresh=" + str(te_thresh)) 
    plt.rcParams.update({'font.size': 26})
    plt.figtext(0.02, 0.02, textstr, fontsize=10)
    plt.savefig(str(paths_dir + edge_type + "_te-" + str(te_thresh) + "_paths.jpg"))
    plt.clf() 



def strongest_path(tree,graph,root):
    '''
    returns a list of edges to color
    '''
    pathway = []
    # Get layers from tree
    edges_from_source = tree.out_edges(root)
    #print(edges_from_source)
    out_edges = {}
    for i in edges_from_source:
        #m,n = graph.get_edge_data(*i).items()
        #print(list(graph.get_edge_data(*i).items()))
        n = list(graph.get_edge_data(*i).values())
        out_edges[i]=n
    #print(out_edges)
    max_val = max(out_edges.values())
    selection = (k for k, v in out_edges.items() if v == max_val)
    #print("selection",*selection)
    elem = [*selection]
    f,t = elem[0]
    pathway.append((f,t))
    #print(pathway)
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
        #print(pathway)
        in_node = t
    return pathway


def plot_htrees(graphs, tree_dir, edge_type,te_thresh, actors, visited_lim, depth_lim):
    for root, graph in graphs.items():

        if not graph.has_node(root):
            return
        
        tree_edges = list(graph.edges)
        tree = bfs_tree_AB(G=graph, source=root, visited_lim=visited_lim, depth_lim = depth_lim, edges = tree_edges)
        nx.relabel_nodes(tree,actors,copy=False)
        nx.relabel_nodes(graph,actors,copy=False)
        root = actors[root]
        colormap_nodes = []
        for node in tree:
            #if tree.out_degree(node)==0:
            if graph.out_degree(node)==0:    
                colormap_nodes.append('green')
            else: 
                colormap_nodes.append('#1f78b4')

        # Color edges in strongest TE path
        #p = tree.get_edge_data(12,7)
        #print("weight:", p)

        pathway = strongest_path(tree,graph,root)
        print(pathway)
        pos = graphviz_layout(tree, prog='dot', args="-Grankdir=LR")
        nx.draw(tree, pos, node_color=colormap_nodes, with_labels=True, font_size=16, node_size=450)
        #nx.draw(tree, pos, node_color=colormap_nodes, edge_color=colormap_edges, with_labels=True, font_size=16, node_size=450)
        plt.figure(3,figsize=(17,50))
        #TODO - fix node_type = ['Continued', 'Terminal', 'Unexpanded, not terminal']
        # fix: plt.legend(handles = node_type, loc = 'lower left')
        # TODO Print weight labels on edges
        #weight_labels = nx.get_edge_attributes(tree,'weight')
        #nx.draw_networkx_edge_labels(tree,pos,edge_labels=weight_labels)
        #nx.draw_networkx_edge_labels(graph,
        #                     pos,
        #                     edge_labels={(u, v): d for u, v, d in graph.edges(data="weight")},
        #                     label_pos=.66)
        plt.savefig(str(tree_dir + edge_type + "-te-" + str(te_thresh) + "-root-"+ str(root) + '-tree.jpg'))
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
            #edges_from_this_level = in_edges_df[in_edges_df['Source'].isin(this_level_nodes)]
            e = in_edges_df[in_edges_df['Source'].isin(this_level_nodes)]
            #print(e) 
            # Replace edge to visited node with new edge to new terminal node with same ID
            for index, edge in e.iterrows():
                #print(edge)
                from_node = edge['Source']
                to_node = edge['Target']
                if visited[to_node]>0:

                    #add new edge to new node with same outgoing actor ID
                    new_node = 120 + n
                    n+=1
                    actor_name = actors[to_node]
                    

                    #actors[new_node] = str(actor_name + '_NE') # for not expanded #TODO THIS LINE BREAKS LAYER EDGES
                    #print(actors)


                    nodepos = ((e['Source']==from_node) & (e['Target']==to_node))
                    e.loc[nodepos, ['Target']]=new_node
                    #print(e.loc[nodepos])
                    #print(actors)
                    # actors.append
                    
                    #somehow assign a new color/shape
            #print(e)
            visited_cap = set([k for k, v in visited.items() if v > max_visits])
            #e = edges_from_this_level[~edges_from_this_level['Target'].isin(visited_cap)]
            e = e[~e['Target'].isin(visited_cap)]

           
            root_df = root_df.append(e, ignore_index=True)
            #this_level_nodes = set(edges_from_this_level['Target'].to_list()).difference(visited_cap)
            this_level_nodes = set(e['Target'].to_list()).difference(visited_cap)
            #this_level_nodes = e['Target'].to_list()
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
    #print(list(edges_gen)) THIS BREAKS THINGS
    #TODO
    # Only include edges from one layer to next
    # Maybe just do something like...
    # edges_gen = edges from te_rollout
    #T.add_edges_from(edges_gen)
    T.add_edges_from(edges)
    return T


def generic_bfs_edges_AB(G, source, visited_lim, depth_lim, neighbors=None, depth_limit=None, sort_neighbors=None):
    if callable(sort_neighbors):
        _neighbors = neighbors
        neighbors = lambda node: iter(sort_neighbors(_neighbors(node)))

    #visited = {source}
    #visited_edges = {}
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
