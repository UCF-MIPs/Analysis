import networkx as nx


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

