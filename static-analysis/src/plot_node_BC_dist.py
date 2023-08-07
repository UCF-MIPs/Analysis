import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_node_BC_dist(graph_dict):
    if not os.path.exists('./bc_plots/nodes/'):
        os.makedirs('./bc_plots/nodes/')

    for edge_type, te_dict in graph_dict.items():
        for te_thresh, graph in te_dict.items():
            # Compute betweenness centrality for each node
            bc_node = nx.betweenness_centrality(graph, normalized=False)
            if not bc_node:
                print(f'The graph for {edge_type} at threshold {te_thresh} has no nodes. Skipping the graph.')
                continue

            min_bc = min(bc_node.values())
            max_bc = max(bc_node.values())

            #bc_node_list = list(bc_node.values())
            if max_bc != min_bc:
                bc_node_list = [(val - min_bc) / (max_bc - min_bc) for val in bc_node.values()]
            else:
                bc_node_list = [0 for _ in bc_node.values()]

            if bc_node_list:
                # Compute the mean and std deviation
                bc_node_mean = sum(bc_node_list) / len(bc_node_list)
                bc_node_std = (sum((i - bc_node_mean) ** 2 for i in bc_node_list) / len(bc_node_list)) ** 0.5
                
                plt.grid(True, alpha = 0.3)
                plt.figure(figsize=(12,6))
                sns.histplot(bc_node_list, bins=20, kde=True)
                plt.title(f'{edge_type} Node Betweenness Centrality Distribution')
                plt.xlabel('Betweenness Centrality')
                plt.ylabel('Density')
                plt.text(0.85, 0.9, f'Mean: {bc_node_mean:.6f}\nStd: {bc_node_std:.6f}', transform=plt.gca().transAxes)
                plt.text(0.85, 0.85, f'TE Threshold: {te_thresh}', transform=plt.gca().transAxes)
                fig_name = f'{edge_type}_{str(te_thresh)}_bc_distribution.png'
                plt.savefig(f'./bc_plots/nodes/{fig_name}')
        
                plt.tight_layout()
                plt.close()
