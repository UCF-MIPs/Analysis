def generate_edge_types():
    #capture all edge types
    from_edges = ['UF', 'UM', 'TF', 'TM']
    to_edges   = ['UF', 'UM', 'TF', 'TM']

    #edge_types = ['UF_TM','UF_TM'] # repeat a few due to blank output #TODO fix
    edge_types = []

    for i in from_edges:
        for j in to_edges:
            edge_types.append(f"{i}_{j}")

    edge_types.append('total_te')

    return edge_types
