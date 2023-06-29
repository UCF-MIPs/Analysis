
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


