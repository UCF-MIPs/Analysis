import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import os

def plot_line_hist(graph_dict, data_name):

    if not os.path.exists('./aggregates/'):
        os.makedirs('./aggregates/')

    color_dict = {'UF_UF': '#1ABC9C', 'UM_UM': '#1ABC9C', 'TF_TF': '#1ABC9C', 'TM_TM': '#1ABC9C',
                  'TM_TF': '#F1C40F', 'TF_TM': '#F1C40F', 'UM_UF': '#F1C40F', 'UF_UM': '#F1C40F',
                  'TM_UM': '#E84393', 'UM_TM': '#E84393', 'TF_UF': '#E84393', 'UF_TF': '#E84393',
                  'TM_UF': '#27AE60', 'UF_TM': '#27AE60', 'TF_UM': '#27AE60', 'UM_TF': '#27AE60'}

    quadrant_titles = ["Echochamber", "Audience Crossover", "Credibility Crossover", "Credibility and Audience Crossover"]
    quadrant_cells = [["TM_TM", "TF_TF", "UM_UM", "UF_UF"],
                  ["TM_TF", "TF_TM", "UM_UF", "UF_UM"],
                  ["TM_UM", "TF_UF", "UM_TM", "UF_TF"],
                  ["TM_UF", "TF_UM", "UM_TF", "UF_TM"]]

    all_thresholds = list(set([thresh for edges in graph_dict.values() for thresh in edges.keys()]))
    all_thresholds.sort()
    plt.style.use('seaborn')

    for te_thresh in all_thresholds:
        fig, axs = plt.subplots(4, 4, figsize=(16, 14), sharey=True)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.96, wspace=0.2, hspace=0.45)

        for i in range(4):
            fig.text(0.5, 0.98 - i * 0.23, quadrant_titles[i], ha='center', fontsize=18, fontweight='bold', transform=fig.transFigure)
            for j in range(4):
                cell_title = quadrant_cells[i][j]
                axs[i, j].set_title(cell_title, fontsize = 15, fontweight='bold', y=0.8)

                graph = graph_dict[cell_title].get(te_thresh, nx.DiGraph())
                bc_values = nx.betweenness_centrality(graph, normalized=False).values()  # compute betweenness centrality
                df = pd.DataFrame({"Betweenness Centrality": bc_values, "Edge Type": [cell_title]*len(bc_values)})
                df["Betweenness Centrality"] = (df["Betweenness Centrality"] - df["Betweenness Centrality"].min()) / (df["Betweenness Centrality"].max() - df["Betweenness Centrality"].min())
                
                try:
                    if not df.empty:
                        axs[i, j].grid(True, alpha=0.5)
                        sns.histplot(data=df, bins=20, x="Betweenness Centrality", hue="Edge Type", kde=True, 
                                        palette=color_dict, ax=axs[i, j], legend=False)
                    else:
                        axs[i, j].text(0.4, 5, 'N/A', fontsize=22)
                        print(f"No data for {cell_title} at threshold {te_thresh}, skipping plot.")
                        continue

                    if j == 0:
                        axs[i, j].set_ylabel('')
                    axs[i, j].set_xlabel('')
                except np.linalg.LinAlgError:
                    print(f"Skipping plot for cell {cell_title} due to singular matrix error.")
                    continue

        fig.text(0.4, 0.06, 'Betweenness Centrality Values', va='center', rotation='horizontal', fontsize=16)
        fig.text(0.01, 0.5, 'Count', va='center', rotation='vertical', fontsize=16)
        fig.text(0.15, 0.02, f'Threshold: {te_thresh}        Data: {data_name}       Plot Type: Betweenness Centrality Histogram', va='center', rotation='horizontal', fontsize=20)
        fig_name = f'line_aggr_bc_thresh_{te_thresh}.png'
        plt.savefig(f'./aggregates/{fig_name}')
