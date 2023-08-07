import networkx as nx

def compute_basic_metrics(graph_dict):

    results = {}

    # Loop over all edge types and thresholds in graph_dict
    for edge_type, graph_list in graph_dict.items():
        results[edge_type] = {}
        for threshold, graph in graph_list.items():
            
            # calculate in-degree and out-degree for each node
            in_degrees = dict(graph.in_degree())
            out_degrees = dict(graph.out_degree())
            
            # calculate total in-degree and out-degree for the network
            total_in_degree = sum(in_degrees.values())
            total_out_degree = sum(out_degrees.values())
            
            # calculate total number of nodes and edges in the network
            total_nodes = graph.number_of_nodes()
            total_edges = graph.number_of_edges()
            
            # calculate node betweenness centrality
            node_betweenness = nx.betweenness_centrality(graph)
            
            # calculate edge betweenness centrality
            edge_betweenness = nx.edge_betweenness_centrality(graph)

            results[edge_type][threshold] = {
                'total_in_degree': total_in_degree,
                'total_out_degree': total_out_degree,
                'in_degrees': in_degrees,
                'out_degrees': out_degrees,
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'node_betweenness': node_betweenness,
                'edge_betweenness': edge_betweenness
            }

    return results
