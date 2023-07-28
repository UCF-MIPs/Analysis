import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import networkx as nx
import os

def plot_total_nodes_edges_networks(graph_dict, data_name):

    if not os.path.exists('./totals/'):
        os.makedirs('./totals/')

    # Get the unique thresholds from all edge types
    unique_thresholds = set()
    for edge_type, threshold_dict in graph_dict.items():
        unique_thresholds.update(threshold_dict.keys())
    
    # Sort the unique thresholds
    unique_thresholds = sorted(list(unique_thresholds))
    plt.style.use("seaborn")

    # Define the edge titles and corresponding edge types
    quadrant_titles = ["Echochamber", "Audience Crossover", "Credibility Crossover", "Credibility and Audience Crossover"]
    edge_types_grouped = [["TM_TM", "TF_TF", "UM_UM", "UF_UF"],
                  ["TM_TF", "TF_TM", "UM_UF", "UF_UM"],
                  ["TM_UM", "TF_UF", "UM_TM", "UF_TF"],
                  ["TM_UF", "TF_UM", "UM_TF", "UF_TM"]]

    # Create an array for each edge type
    edge_types = [edge for group in edge_types_grouped for edge in group]
    edge_indices = np.arange(len(edge_types))

    for threshold in unique_thresholds:
        total_nodes = []
        total_edges = []

        # Collect data for plotting
        for edge_type in edge_types:
            graph = graph_dict.get(edge_type, {}).get(threshold, nx.DiGraph())
            total_nodes.append(graph.number_of_nodes())
            total_edges.append(graph.number_of_edges())

        # Create a new figure
        plt.figure(figsize=(15, 8))

        # Set the width of the bars
        bar_width = 0.35

        # Plot the total number of nodes and edges per edge type
        plt.bar(edge_indices - bar_width/2, total_nodes, bar_width, label='Total Nodes')
        plt.bar(edge_indices + bar_width/2, total_edges, bar_width, label='Total Edges')

        plt.yscale('log')
        plt.xticks(edge_indices, edge_types, rotation = 45)
        plt.xlabel('Edge Types')
        plt.ylabel('Count')
        plt.grid(True)

        # Add a legend to the plot
        legend1 = plt.legend(['Total Nodes', 'Total Edges'])

        # Add the legend manually to the current Axes.
        plt.gca().add_artist(legend1)

        # Create another legend for the second legend at the bottom
        dummy_handle = mpatches.Patch(color='none')
        legend2 = plt.legend([dummy_handle], [f'Threshold: {threshold}   Data: {data_name}   Plot: Total Number of Nodes and Edges Per Network'], 
                             handlelength=0, handletextpad=0, loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, fontsize = 13)
        
        for item in legend2.legendHandles:
            item.set_visible(False)

        # Add quadrant titles to the plot
        for i in range(4):
            plt.text(i * 4 + 1.5, plt.gca().get_ylim()[1], quadrant_titles[i], ha='center', va='bottom', fontsize = 14)

        # Add vertical dashed lines to the plot
        for i in range(1, 4):
            plt.axvline(x=i*4-0.5, color='black', linestyle='dashed')

        plt.tight_layout()

        # Save the figure to a file
        plt.savefig(f'./totals/thresh_{threshold}_total_nodes_edges.png')

        # Close the figure to free up memory
        plt.close()
