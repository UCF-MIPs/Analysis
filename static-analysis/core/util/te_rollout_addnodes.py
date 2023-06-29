import pandas as pd
import networkx as nx 


def te_rollout_addnodes(in_roots, in_edges_df, max_visits, actors):
    # number of added nodes
    #TODO have appended number in unexpanded node names represent
    # number of times it shows up, instead of random value
    n=0
    te_levels_df = {}
    all_root_dfs = {}
    for in_root in in_roots:
        visited = {}
        root_df = pd.DataFrame()
        for node in range(10000):
            visited.update({node:0})
        this_level_nodes = in_root
        te_values = []
        this_level = 0
        while True:
            if(this_level==0):
                this_level_nodes = [this_level_nodes]
            last_visited = visited.copy()
            for node in this_level_nodes:
                visited[node] += 1
                print(node)
            if(last_visited == visited):
                break
            this_level += 1
            e = in_edges_df[in_edges_df['Source'].isin(this_level_nodes)]
            # Replace edge to visited node with new edge to new terminal node with same ID
            for index, edge in e.iterrows():
                from_node = edge['Source']
                to_node = edge['Target']
                if visited[to_node]>0:
                    #add new edge to new node with same outgoing actor ID
                    new_node = 120 + n
                    n+=1
                    actor_name = actors[to_node]
                    actors[new_node] = f"{actor_name}_{n}"
                    nodepos = ((e['Source']==from_node) & (e['Target']==to_node))
                    e.loc[nodepos, ['Target']]=new_node
            visited_cap = set([k for k, v in visited.items() if v > max_visits])
            e = e[~e['Target'].isin(visited_cap)]
            root_df = root_df.append(e, ignore_index=True)
            this_level_nodes = set(e['Target'].to_list()).difference(visited_cap)
        all_root_dfs.update({in_root:root_df})

    return all_root_dfs, actors

