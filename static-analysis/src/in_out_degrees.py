import networkx as nx

def compute_degrees(graph):

    """ 
    This function calculates the in and out degrees of a network, their total and averages.
    It inputs the graph and returns the dictionary of all values.
    
    """
    # check if the graph is directed
    if not graph.is_directed():
        raise ValueError("G must be a directed graph.")
        
    # calculate in-degree and out-degree for each node
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    
    # calculate total in-degree and out-degree for the network
    total_in_degree = sum(in_degrees.values())
    total_out_degree = sum(out_degrees.values())
    
    # calculate average in-degree and out-degree for the network
    average_in_degree = total_in_degree / len(graph)
    average_out_degree = total_out_degree / len(graph)

    result = {
        'total_in_degree': total_in_degree,
        'total_out_degree': total_out_degree,
        'average_in_degree': average_in_degree,
        'average_out_degree': average_out_degree,
        'in_degrees': in_degrees,
        'out_degrees': out_degrees
    }
    return result
