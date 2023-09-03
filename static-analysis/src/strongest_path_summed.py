import networkx as nx

def strongest_path_summed(tree, graph, root): # derived from modification of Alex Baekey's strongest_path_summed method
    '''
    returns a list of edges to color
    path selected by summed TE over all possible paths
    '''

    pathways = []
    roots = []
    leaves = []

    # classify nodes in tree as root nodes or leaf nodes
    for node in tree.nodes:
        if tree.in_degree(node) == 0:
            roots.append(node)
        elif tree.out_degree(node) == 0 :
            leaves.append(node)

    # identify all simple paths from each root node to each leaf node
    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_edge_paths(tree, root, leaf) :
                pathways.append(path)

    # traverse each previously identified path, and find the total of all edge weights for each path
    total = 0
    total_temp=0
    for path in pathways:
        for edge in path:
            # in case this is an aliased edge added during tree construction, dealias the edge so it will refer to the corresponding edge in graph; if the edge is not aliased, this will change nothing
            edge_dealiased = (edge[0].rstrip(), edge[1].rstrip())
            total_temp += float(list(graph.get_edge_data(*edge_dealiased).values())[0])
        if total_temp > total:
            total = total_temp
            strongest_pathway = path
        total_temp=0
    return strongest_pathway, total


