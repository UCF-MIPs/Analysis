import matplotlib.pyplot as plt
import networkx as nx
import os

def plot_network_BC(graph_dict, actors, data_name):

    if not os.path.exists('./bc_plots/graphs/'):
        os.makedirs('./bc_plots/graphs/')

    plt.style.use("classic")
    plt.grid('off')

    for edge_type, threshold_dict in graph_dict.items():
        for te_thresh, G in threshold_dict.items():
            if len(G) == 0 or G.number_of_edges() == 0:
                print(f'The graph for {edge_type} at threshold {te_thresh} has no nodes or edges. Skipping this graph.')
                continue
            G1 = nx.relabel_nodes(G, actors)
            measures = nx.betweenness_centrality(G1, normalized=False)

             # Generate the position layout
            pos = nx.spiral_layout(G1)

            # Normalize the measures values to range between 100 and 1000
            if max(measures.values()) == min(measures.values()):
                norm_measures = [500 for _ in measures.values()]  # If all measures are the same, set a constant node size
            else:
                norm_measures = [100 + 900 * ((m - min(measures.values())) / (max(measures.values()) - min(measures.values()))) for m in measures.values()]

            node_colors = list(norm_measures)

            plt.figure(figsize=(12,10))  # Increasing the size of the plot
            plt.margins(0.1)  # Adding margins to the plot

            nodes = nx.draw_networkx_nodes(G1, pos, node_size=norm_measures, cmap=plt.cm.plasma, 
                                           node_color=node_colors,
                                           nodelist=G1.nodes())

            nx.draw_networkx_labels(G1, pos)
            nx.draw_networkx_edges(G1, pos)

            plt.title(f'{edge_type} Network ', fontsize = 14)
            #plt.tight_layout()
            plt.text(0, -0.05, f'Threshold: {te_thresh}        Data: {data_name}       Plot Type: Influential Nodes Based on the Betweenness Centrality ', va='center', rotation='horizontal', fontsize = 12, transform=plt.gca().transAxes)

            cbar = plt.colorbar(nodes)
            cbar.set_label('Normalized Betweenness Centrality Values', rotation=270, labelpad=15)
            
            fig_name = f'{edge_type}_{str(te_thresh)}_BC_centrality.png'
            plt.savefig(f'./bc_plots/graphs/{fig_name}')
            plt.close()
