import networkx as nx

def compute_degrees_nodes_edges(graph):

    # calculate in-degree and out-degree for each node
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    
    # calculate total in-degree and out-degree for the network
    total_in_degree = sum(in_degrees.values())
    total_out_degree = sum(out_degrees.values())
    
    # calculate average in-degree and out-degree for the network
    average_in_degree = total_in_degree / len(graph)
    average_out_degree = total_out_degree / len(graph)

    # calculate total number of nodes and edges in the network
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()

    result = {
        'total_in_degree': total_in_degree,
        'total_out_degree': total_out_degree,
        'average_in_degree': average_in_degree,
        'average_out_degree': average_out_degree,
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'total_nodes': total_nodes,
        'total_edges': total_edges
    }
    return result

