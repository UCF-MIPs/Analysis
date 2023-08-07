import networkx as nx

def get_top_n_by_bc(graph_dict, actors, n):
    results = {}
    results1 = {}

    # Process each graph type in graph_dict
    for edge_type, graph_list in graph_dict.items():
        results[edge_type] = {}
        results1[edge_type] = {}
        # Process each graph in the graph list
        for threshold, graph in graph_list.items():
            # Relabel nodes
            g = nx.relabel_nodes(graph, actors)
            
            # Calculate betweenness centrality
            centrality = nx.betweenness_centrality(g, normalized= False)

            # Sort nodes by betweenness centrality and retrieve top n percent
            sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
            top_n_percent = sorted_centrality[:max(1, int(len(sorted_centrality)*n))]
            #print(top_n_percent)
            # Calculate the average betweenness centrality of top n percent
            if len(top_n_percent) == 0:
                avg_centrality = 0
            else:
                avg_centrality = sum(val for node, val in top_n_percent) / len(top_n_percent)
            
            results[edge_type][threshold] = avg_centrality
            results1[edge_type][threshold] = top_n_percent

    return results1
