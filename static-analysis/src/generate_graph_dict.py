import networkx as nx

def generate_graph_dict(df, te_thresholds, edge_types):
    graph_dict = {}
    for edge_type in edge_types:
        graph_dict[edge_type] = {}
        for te_thresh in te_thresholds:
            graph_df = df.loc[(df[edge_type] > te_thresh)]
            g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
            graph_dict[edge_type][te_thresh] = g
    return graph_dict
    
