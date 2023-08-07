import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_edge_BC_dist(graph_dict):
    if not os.path.exists('./bc_plots/edges/'):
        os.makedirs('./bc_plots/edges/')

    for edge_type, te_dict in graph_dict.items():
        for te_thresh, graph in te_dict.items():
            bc_edge = nx.edge_betweenness_centrality(graph, normalized=False)
            
            if not bc_edge:
                print(f'The graph for {edge_type} at threshold {te_thresh} has no edges. Skipping the graph.')
                continue

            min_bc = min(bc_edge.values())
            max_bc = max(bc_edge.values())

            #bc_edge_list = list(bc_edge.values())
            if max_bc != min_bc:
                bc_edge_list = [(val - min_bc) / (max_bc - min_bc) for val in bc_edge.values()]
            else:
                bc_edge_list = [0 for _ in bc_edge.values()]
            
            if bc_edge_list:
                # Compute the mean and std deviation
                bc_edge_mean = sum(bc_edge_list) / len(bc_edge_list)
                bc_edge_std = (sum((i - bc_edge_mean) ** 2 for i in bc_edge_list) / len(bc_edge_list)) ** 0.5

                plt.grid(True, alpha = 0.3)
                plt.figure(figsize=(12,6))
                sns.histplot(bc_edge_list, bins=20, kde=True)
                plt.title(f'{edge_type} Edge Betweenness Centrality Distribution')
                plt.xlabel('Betweenness Centrality')
                plt.ylabel('Density')
                plt.text(0.85, 0.9, f'Mean: {bc_edge_mean:.6f}\nStd: {bc_edge_std:.6f}', transform=plt.gca().transAxes)
                plt.text(0.85, 0.85, f'TE Threshold: {te_thresh}', transform=plt.gca().transAxes)
                fig_name = f'{edge_type}_{str(te_thresh)}_bc_distribution.png'
                plt.savefig(f'./bc_plots/edges/{fig_name}')
        
                plt.tight_layout()
                plt.close()