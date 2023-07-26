import pandas as pd
import networkx as nx 

def te_rollout(in_root, in_edges_df, max_visits):
    '''
    Generates layer-wise propogation of edges in a network, 
    avoids cycles with a duplicate node renaming scheme

        Parameters:

            in_root (str): name of root node

            in_edges_df (Pandas DataFrame): df of original network

            max_visits (int): Max number of times any node can be visited

        Returns:

            root_df (Pandas DataFrame): df created from edges from a certain root node

    '''
    n=0
    visited = {}
    root_df = pd.DataFrame()
    for node in pd.unique(in_edges_df[['Source', 'Target']].values.ravel('K')):
        visited.update({node:0})
    this_level_nodes = [in_root]
    while True:
        last_visited = visited.copy()
        for node in this_level_nodes:
            visited[node] += 1
        if(last_visited == visited):
            break
        e = in_edges_df[in_edges_df['Source'].isin(this_level_nodes)]
        for index, edge in e.iterrows():
            from_node = edge['Source']
            to_node = edge['Target']
            if visited[to_node]>0:
                #new_node = f"{to_node}_{visited[to_node]}"
                new_node = f"{to_node}_{n}"
                #visited[to_node] +=1
                n += 1
                #visited.update({new_node:(visited[to_node]+1)})
                visited.update({new_node:0})
                nodepos = ((e['Source']==from_node) & (e['Target']==to_node))
                e.loc[nodepos, ['Target']] = new_node
        visited_cap = set([k for k, v in visited.items() if v > max_visits])
        e = e[~e['Target'].isin(visited_cap)]
        root_df = root_df.append(e, ignore_index=True)
        this_level_nodes = set(e['Target'].to_list()).difference(visited_cap)
    return root_df
    
