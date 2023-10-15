import networkx as nx

def graph_dict2metric_dict(graph_dict, metric_selection):
    metric_dict = {}
    for edge_type, threshold_graphs in graph_dict.items():
        metric_dict[edge_type] = {}
        for te_threshold, graph in threshold_graphs.items():
            # Calculate the betweenness centrality values
            bc_values = nx.betweenness_centrality(graph, normalized=False)
            # Calculate the total out-degree
            out_degree_sum = sum([out_degree for node, out_degree in graph.out_degree()])
            # Store the metrics in a dictionary
            if metric_selection=='outdegree':
                metrics = {'total_out_degree': out_degree_sum}
            elif metric_selection=='bc':
                metrics = {'betweenness_centrality_values': bc_values}

            # Save the metrics in the overall metrics dictionary
            metric_dict[edge_type][te_threshold] = metrics
    return metric_dict
