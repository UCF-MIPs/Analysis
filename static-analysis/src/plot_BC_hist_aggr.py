import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import os

def plot_BC_hist_aggr(graph_dict):

    if not os.path.exists('./bc_plots/'):
        os.makedirs('./bc_plots/')
    
    color_dict = {'UF': '#1ABC9C', 'UM': '#F1C40F', 'TF': '#E84393', 'TM': '#27AE60', 'to' : '#E74C3C'}
    
    # Let's first iterate over each unique threshold
    thresholds = next(iter(graph_dict.values())).keys()  # Assuming same thresholds for each edge type
    plt.grid(True, alpha = 0.3)

    for threshold in thresholds:
        # For each threshold, create a new set of subplots
        fig, axs = plt.subplots(5, 4, figsize=(10, 10))
        axs = axs.flatten()
        
        i = 0
        # Now iterate over each edge type and corresponding sub-dictionary in graph_dict
        for edge_type, sub_dict in graph_dict.items():
            # Get the graph corresponding to the current threshold
            G = sub_dict[threshold]
            
            # Calculate betweenness centrality for each node in the current graph
            betweenness = nx.betweenness_centrality(G, normalized=False)
            
            # Convert betweenness values to a DataFrame
            df = pd.DataFrame({"Betweenness": list(betweenness.values())})
            
            # Create histogram
            sns.histplot(df["Betweenness"], ax=axs[i], color=color_dict[edge_type[:2]], kde=True, log_scale=(False, False))
            
            axs[i].set_title(f'{edge_type}', fontsize=11, fontweight='bold')
            axs[i].set_xlabel('Values', fontsize=8)
            axs[i].set_ylabel('Count', fontsize=8)
            i += 1

        # Remove any unused subplots
        while i < len(axs):
            fig.delaxes(axs[i])
            i += 1

        fig.suptitle(f'Histograms of Betweenness Centrality Values for Networks at Threshold {threshold}')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.6, wspace=0.3)
        plt.savefig(f'./bc_plots/thresh_{threshold}_bc_hist_aggr.png')
        # plt.show()  # Uncomment this to show plots inline


