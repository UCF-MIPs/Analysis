def strongest_path_greedy(tree,graph,root): # derived from modification of Alex Baekey's strongest_path_greedy method
    '''
    returns a list of edges to color
    path selected by next strongest edge
    '''
    pathway = []
    total_strength = 0
    # Get layers from tree
    edges_from_source = tree.out_edges(root)
    out_edges = {}
    for i in edges_from_source:
        # in case this is an aliased edge added during tree construction, dealias the edge so it will refer to the corresponding edge in graph; if the edge is not aliased, this will change nothing
        i_dealiased = (i[0].rstrip(), i[1].rstrip())
        n = list(graph.get_edge_data(*i_dealiased).values())
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
            # in case this is an aliased edge added during tree construction, dealias the edge so it will refer to the corresponding edge in graph; if the edge is not aliased, this will change nothing
            i_dealiased = (i[0].rstrip(), i[1].rstrip())
            n = list(graph.get_edge_data(*i_dealiased).values())
            out_edges[i]=n
        max_val = max(out_edges.values())
        total_strength += max_val[0]
        selection = (k for k, v in out_edges.items() if v== max_val)
        elem = [*selection]
        f,t = elem[0]
        pathway.append((f,t))
        in_node = t
    return pathway, total_strength

