import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D 

def plot_total_nodes_networks(graph_dict, data_name):
    result_dict = {}

    # calculate total number of nodes and edges from graphs in graph_dict
    for edge_type, threshold_dict in graph_dict.items():
        result_dict[edge_type] = {}
        for threshold, graph in threshold_dict.items():
            result_dict[edge_type][threshold] = {
                'total_nodes': len(graph.nodes()),
            }

    # Get the unique thresholds from all edge types
    unique_thresholds = set()
    for edge_type, threshold_dict in result_dict.items():
        unique_thresholds.update(threshold_dict.keys())

    # Sort the unique thresholds
    unique_thresholds = sorted(list(unique_thresholds))

    color_dict = {'UF_UF': '#1ABC9C', 'UM_UM': '#1ABC9C', 'TF_TF': '#1ABC9C', 'TM_TM': '#1ABC9C',
                  'TM_TF': '#F1C40F', 'TF_TM': '#F1C40F', 'UM_UF': '#F1C40F', 'UF_UM': '#F1C40F',
                  'TM_UM': '#E84393', 'UM_TM': '#E84393', 'TF_UF': '#E84393', 'UF_TF': '#E84393',
                  'TM_UF': '#27AE60', 'UF_TM': '#27AE60', 'TF_UM': '#27AE60', 'UM_TF': '#27AE60'}

    quadrant_titles = ["Echochamber", "Audience Crossover", "Credibility Crossover", "Credibility and Audience Crossover"]
    edge_types = [["TM_TM", "TF_TF", "UM_UM", "UF_UF"],
                  ["TM_TF", "TF_TM", "UM_UF", "UF_UM"],
                  ["TM_UM", "TF_UF", "UM_TM", "UF_TF"],
                  ["TM_UF", "TF_UM", "UM_TF", "UF_TM"]]

    all_edge_types = [edge for quadrant in edge_types for edge in quadrant]
    
    bar_width = 0.7
    plt.style.use("seaborn")
    for threshold in unique_thresholds:
        fig, ax = plt.subplots(figsize=(10,6))
        total_nodes = []
        colors = []
        # Collect data for plotting
        for edge_type in all_edge_types:
            threshold_dict = result_dict.get(edge_type, {})
            metrics = threshold_dict.get(threshold, {'total_nodes': 0})
            total_nodes.append(metrics['total_nodes'])
            colors.append(color_dict[edge_type])

        # Plot the total number of nodes and edges per edge type
        ax.bar(range(len(all_edge_types)), total_nodes, bar_width, color=colors)

        #ax.set_yscale('log')
        plt.xticks(rotation = 45)
        ax.set_xlabel('Edge Types')
        ax.set_ylabel('Count')
        ax.grid(True)

        # Set the x-tick labels
        xticks_locs = np.arange(len(all_edge_types))
        ax.set_xticks(xticks_locs)
        ax.set_xticklabels(all_edge_types, rotation=45, ha="right")

        # Create legend
        quadrant_colors = [color_dict[edge_type[0]] for edge_type in edge_types]
        legend_patches = [Line2D([0], [0], color=color, marker='D', linestyle='None') for color in quadrant_colors]
        
        legend1 = plt.legend(handles=legend_patches, loc='lower center', ncol=len(legend_patches), 
                             labels=quadrant_titles, bbox_to_anchor=(0.5, -0.28))  # Update this line

        # Add the legend manually to the current Axes.
        ax.add_artist(legend1)

        # Create another legend for the second legend at the bottom
        dummy_handle = mpatches.Patch(color='none')
        legend2 = plt.legend([dummy_handle], [f'Threshold: {threshold}   Data: {data_name}   Plot: Total Number of Nodes Per Network'], 
                             handlelength=0, handletextpad=0, loc='lower center', bbox_to_anchor=(0.5, -0.38), fancybox=True, shadow=True, fontsize = 13)

        for item in legend2.legendHandles:
            item.set_visible(False)

        plt.tight_layout()

        # Save the figure to a file
        plt.savefig(f'./totals/thresh_{threshold}_total_nodes.png')

        # Close the figure to free up memory
        plt.close()
