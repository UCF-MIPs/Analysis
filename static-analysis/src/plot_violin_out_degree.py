import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

def plot_violin_out_degree(graph_dict):

    color_dict = {'UF': '#1ABC9C', 'UM': '#F1C40F', 'TF': '#E84393', 'TM': '#27AE60', 'to' : '#E74C3C'}

    all_data = []

    for edge_type, threshold_dict in graph_dict.items():
        for te_thresh, graph in threshold_dict.items():
            degrees = [d for n, d in graph.out_degree()]
            color_mapping = [edge_type[:2]]*len(degrees)
            all_data.append((te_thresh, degrees, edge_type, color_mapping))

    all_data = [item for item in all_data if item[2] != 'total_te']

    # get unique thresholds
    thresholds = set([item[0] for item in all_data])

    for te_thresh in thresholds:
        degree_values = []
        all_edge_types = []
        color_mapping = []

        # gather data for this threshold
        for item in all_data:
            if item[0] == te_thresh:
                degree_values.extend(item[1])
                all_edge_types.extend([item[2]]*len(item[1]))
                color_mapping.extend(item[3])
        
        df = pd.DataFrame({"Degree": degree_values, "Edge Type": all_edge_types, "Color": color_mapping})
        
        plt.grid(True, alpha = 0.3)
        plt.figure(figsize=(20, 5))
        sns.violinplot(x="Edge Type", y="Degree", hue="Color", data=df, palette=color_dict, dodge=False)
        plt.title(f'Out Degree by Network Type with Threshold {te_thresh}')
        plt.ylabel('Out Degree Value')
        plt.tight_layout()
        fig_name = f'thresh_{str(te_thresh)}_violin_degree.png'
        plt.savefig(f'./degree_plots/distributions/{fig_name}')
        plt.close()
