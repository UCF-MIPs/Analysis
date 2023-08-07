import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import os

def plot_box_BC(graph_dict, data_name):
    if not os.path.exists('./aggregates/'):
        os.makedirs('./aggregates/')
        
    color_dict = {'UF_UF': '#1ABC9C', 'UM_UM': '#1ABC9C', 'TF_TF': '#1ABC9C', 'TM_TM': '#1ABC9C',
                  'TM_TF': '#F1C40F', 'TF_TM': '#F1C40F', 'UM_UF': '#F1C40F', 'UF_UM': '#F1C40F',
                  'TM_UM': '#E84393', 'UM_TM': '#E84393', 'TF_UF': '#E84393', 'UF_TF': '#E84393',
                  'TM_UF': '#27AE60', 'UF_TM': '#27AE60', 'TF_UM': '#27AE60', 'UM_TF': '#27AE60'}

    all_quadrant_cells = ["TM_TM","TF_TF", "UM_UM", "UF_UF", "TM_TF", "TF_TM", "UM_UF", "UF_UM", "TM_UM", "TF_UF", "UM_TM", "UF_TF", "TM_UF", "TF_UM", "UM_TF", "UF_TM"]

    all_thresholds = list(set([thresh for edges in graph_dict.values() for thresh in edges.keys()]))
    all_thresholds.sort()
    plt.style.use('seaborn')
    plt.yscale('log')

    for te_thresh in all_thresholds:
        fig, ax = plt.subplots(figsize=(18, 6))  # Change plot size

        df = pd.DataFrame()  # initialize dataframe
        for cell_title in all_quadrant_cells:
            graph = graph_dict.get(cell_title, {}).get(te_thresh, nx.DiGraph())
            centrality_values = list(nx.betweenness_centrality(graph, normalized=False).values())
            if centrality_values:  # Only if there is data
                temp_df = pd.DataFrame({"Centrality": centrality_values, "Edge Type": [cell_title]*len(centrality_values)})
                df = pd.concat([df, temp_df])

        # Check if all edge types are represented in the DataFrame. If not, add an empty entry
        for cell_title in all_quadrant_cells:
            if cell_title not in df['Edge Type'].values:
                temp_df = pd.DataFrame({"Centrality": [None], "Edge Type": [cell_title]})
                df = pd.concat([df, temp_df])

        # Convert Edge Type to a categorical variable with the order defined by all_quadrant_cells
        df['Edge Type'] = pd.Categorical(df['Edge Type'], categories=all_quadrant_cells, ordered=True)

        if not df.empty:
            ax.grid(True, alpha=0.8)
            sns.boxplot(x="Edge Type", y="Centrality", data=df, palette=color_dict, ax=ax, order=all_quadrant_cells)

        ax.set_xlabel('', fontsize = 14)
        ax.set_ylabel('Betweenness Centrality Value', fontsize=14)
        ax.set_xticklabels(all_quadrant_cells, fontweight='bold', fontsize=14)
        ax.set_yscale('log')

        # Create custom legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=15),
                           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=15),
                           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=15)]
        ax.legend(legend_elements, [f'Threshold: {te_thresh}', f'Data: {data_name}', 'Plot Type: Betweenness Centrality Box Plots'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5, fontsize = 14)

        # Calculate mean values and add them to the plot
        for i, cell_title in enumerate(all_quadrant_cells):
            mean_val = df[df['Edge Type'] == cell_title]['Centrality'].mean()
            if not pd.isna(mean_val):
                ax.text(i, ax.get_ylim()[0], f'Mean: {mean_val:.2f}', ha='center', va='bottom', fontsize=11)

        fig.subplots_adjust(bottom=0.2)  # adjust the bottom parameter to create space for the legend
        plt.tight_layout()

        fig_name = f'bc_box_thresh_{te_thresh}.png'
        plt.savefig(f'./aggregates/{fig_name}', bbox_inches='tight')  # Add bbox_inches parameter to save the entire image
