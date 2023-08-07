import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.patches import Patch

def plot_violin_out_degree(graph_dict):

    if not os.path.exists('./degree_plots/'):
        os.makedirs('./degree_plots/')

    color_dict = {'UF_UF': '#1ABC9C', 'UM_UM': '#1ABC9C', 'TF_TF': '#1ABC9C', 'TM_TM': '#1ABC9C',
                  'TM_TF': '#F1C40F', 'TF_TM': '#F1C40F', 'UM_UF': '#F1C40F', 'UF_UM': '#F1C40F',
                  'TM_UM': '#E84393', 'UM_TM': '#E84393', 'TF_UF': '#E84393', 'UF_TF': '#E84393',
                  'TM_UF': '#27AE60', 'UF_TM': '#27AE60', 'TF_UM': '#27AE60', 'UM_TF': '#27AE60'}

    quadrant_colors = ['#1ABC9C', '#F1C40F', '#E84393', '#27AE60']
    quadrant_titles = ["Echochamber", "Audience Crossover", "Credibility Crossover", "Credibility and Audience Crossover"]
    
    legend_patches = [Patch(color=color, label=label) for color, label in zip(quadrant_colors, quadrant_titles)]

    quadrant_cells = [["TM_TM", "TF_TF", "UM_UM", "UF_UF"],
                  ["TM_TF", "TF_TM", "UM_UF", "UF_UM"],
                  ["TM_UM", "TF_UF", "UM_TM", "UF_TF"],
                  ["TM_UF", "TF_UM", "UM_TF", "UF_TM"]]

    # flatten the quadrant_cells list for ordering plots
    order_list = [item for sublist in quadrant_cells for item in sublist]

    all_data = []

    plt.style.use('seaborn')
    for edge_type, threshold_dict in graph_dict.items():
        for te_thresh, graph in threshold_dict.items():
            degrees = [d for n, d in graph.out_degree()]
            color_mapping = color_mapping = [edge_type]*len(degrees)
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

        # Create a template DataFrame with all possible edge types excluding 'total_te'
        edge_types_template = pd.DataFrame({"Edge Type": [edge_type for edge_type in graph_dict.keys() if edge_type != 'total_te']})

        # Merge the template DataFrame with the actual DataFrame, filling NaN for missing values
        df = pd.merge(edge_types_template, df, on="Edge Type", how="left")
        
        plt.yscale('log')
        plt.grid(True, alpha = 0.3)
        plt.figure(figsize=(20, 5))
        ax = sns.violinplot(x="Edge Type", y="Degree", hue="Color", data=df, palette=color_dict, dodge=False, order=order_list)
        plt.title(f'Out Degree by Network Type with Threshold {te_thresh}')
        plt.ylabel('Out Degree Value')
        plt.tight_layout()

        # handle legends
        ax.get_legend().remove()  # remove original legend
        plt.legend(handles=legend_patches)  # add custom legend

        fig_name = f'thresh_{str(te_thresh)}_violin_degree.png'
        plt.savefig(f'./degree_plots/{fig_name}')
        plt.close()
