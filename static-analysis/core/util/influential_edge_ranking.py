import networkx as nx

def influential_edge_ranking(g, pulltop=0, edge_names=False):
    '''
    ranking edges in a network based on edge betweenness-centrality
    Returns a list of top edges and the nodes associated with these edges
    If edge_names=False, also returns node names and node betweenness centrality

    returns the list of edges, the list of top nodes within these edges,
    list of nodes and corresponding centralities
    '''
    BC_edges = nx.edge_betweenness_centrality(g, normalized=True)
    sorted_edges = sorted(BC_edges.items(), key=lambda x:x[1], reverse=True)

    top_nodes = set()
    node_centralities = {}

    if pulltop != 0:
        sorted_edges = sorted_edges[:pulltop]
        for edge in sorted_edges:
            top_nodes.update(edge[0])  # edge[0] is a tuple containing the nodes

    if edge_names == True:
        for n, i in enumerate(sorted_edges):
            sorted_edges[n] = i[0]
    else:
        BC_nodes = nx.betweenness_centrality(g, normalized=True)
        for node in top_nodes:
            node_centralities[node] = BC_nodes[node]

    return sorted_edges, list(top_nodes), node_centralities

