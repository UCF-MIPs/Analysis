import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import os


def plot_network_degree_centrality(graph_dict, actors, data_name):

    if not os.path.exists('./degree_plots/graphs/'):
        os.makedirs('./degree_plots/graphs/')
    
    plt.style.use("classic")
    plt.grid('off')
    
    for edge_type, te_dict in graph_dict.items():
        for te_thresh, G in te_dict.items():
            # Check if the graph has nodes
            if len(G) == 0:
                print(f'No nodes in the graph for {edge_type} at threshold {te_thresh}. Skipping this graph.')
                continue

            G1 = nx.relabel_nodes(G, actors)
            measures = nx.out_degree_centrality(G1)

            # Generate the position layout
            pos = nx.spiral_layout(G1)

            # Normalize the measures values to range between 100 and 1000
            if max(measures.values()) == min(measures.values()):
                norm_measures = [500 for _ in measures.values()]  # If all measures are the same, set a constant node size
            else:
                norm_measures = [100 + 900 * ((m - min(measures.values())) / (max(measures.values()) - min(measures.values()))) for m in measures.values()]

            node_colors = list(norm_measures)

            plt.figure(figsize=(12,10))
            plt.margins(0.1)  # Adding margins to the plot

            nodes = nx.draw_networkx_nodes(G1, pos, node_size=norm_measures, cmap=plt.cm.plasma, 
                                        node_color=node_colors,
                                        nodelist=measures.keys())
            #nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))

            nx.draw_networkx_labels(G1, pos)
            nx.draw_networkx_edges(G1, pos)

            plt.title(f'{edge_type} Network ', fontsize = 14)
            #plt.tight_layout()
            plt.text(0, -0.05, f'Threshold: {te_thresh}        Data: {data_name}       Plot Type: Influential Nodes Based on Normalized Degree Centrality ', va='center', rotation='horizontal', fontsize = 12, transform=plt.gca().transAxes)

            cbar = plt.colorbar(nodes)
            cbar.set_label('Normalized Degree Centrality Values', rotation=270, labelpad=15)

            fig_name = f'{edge_type}_{str(te_thresh)}_degree_centr.png'
            plt.savefig(f'./degree_plots/graphs/{fig_name}')
            plt.close()
