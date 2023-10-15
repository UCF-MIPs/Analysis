import networkx as nx


def all_betweenness_centrality(graph):
    """ 
    This function calculates betweenness for all nodes and edges within a network.
    It inputs the generated graph and returns a list of bc values for all edges
    and a list of bc values for all nodes.
    
    """
    node_betweenness_centrality = nx.betweenness_centrality(graph)
    edge_betweenness_centrality = nx.edge_betweenness_centrality(graph)

    # Calculate and print the descriptive statistics for node betweenness centrality
    bc_node_list = list(node_betweenness_centrality.values())
    
    # Calculate and print the descriptive statistics for edge betweenness centrality
    bc_edge_list = list(edge_betweenness_centrality.values())
    return bc_node_list, bc_edge_list
