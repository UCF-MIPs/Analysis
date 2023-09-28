import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import os

def plot_quadrant_bc(graph_dict, data_name):

    if not os.path.exists('./aggregates/'):
        os.makedirs('./aggregates/')
        
    color_dict = {'UF_UF': '#1ABC9C', 'UM_UM': '#1ABC9C', 'TF_TF': '#1ABC9C', 'TM_TM': '#1ABC9C',
                  'TM_TF': '#F1C40F', 'TF_TM': '#F1C40F', 'UM_UF': '#F1C40F', 'UF_UM': '#F1C40F',
                  'TM_UM': '#E84393', 'UM_TM': '#E84393', 'TF_UF': '#E84393', 'UF_TF': '#E84393',
                  'TM_UF': '#27AE60', 'UF_TM': '#27AE60', 'TF_UM': '#27AE60', 'UM_TF': '#27AE60'}

    quadrant_titles = ["Echochamber", "Audience Crossover", "Credibility Crossover", "Credibility and Audience Crossover"]
    quadrant_cells = [["TM_TM", "TF_TF", "TM_TF", "TF_TM"],
                       ["UM_UM", "UF_UF", "UM_UF", "UF_UM"],
                  ["TM_UM", "TF_UF", "TM_UF", "TF_UM"],
                   ["UM_TM", "UF_TF", "UM_TF", "UF_TM"]]

    all_thresholds = list(set([thresh for edges in graph_dict.values() for thresh in edges.keys()]))
    all_thresholds.sort()
    plt.style.use('seaborn')

    for te_thresh in all_thresholds:
        fig, axs = plt.subplots(4, 4, figsize=(12, 12), sharey = True) # create a 4x4 grid of Axes
        fig.subplots_adjust(hspace = 0.6, wspace=0.2, bottom=0.2)  # adjust the spacing and create space for the legend

        # Add big axes and hide frame
        big_ax = fig.add_subplot(111, frame_on=False) 
        big_ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

        for i in range(4):
            for j in range(4):
                cell_title = quadrant_cells[i][j]
                axs[i, j].set_title(cell_title, fontsize = 11, fontweight='bold', y=0.8)

                graph = graph_dict[cell_title].get(te_thresh, nx.DiGraph())
                centrality = nx.betweenness_centrality(graph)
                df = pd.DataFrame({"Betweenness Centrality": list(centrality.values()), "Edge Type": [cell_title]*len(centrality)})
                df["Betweenness Centrality"] = (df["Betweenness Centrality"] - df["Betweenness Centrality"].min()) / (df["Betweenness Centrality"].max() - df["Betweenness Centrality"].min())

                if not df.empty:
                    axs[i, j].grid(True, alpha=0.5)
                    sns.histplot(df, bins = 20, x="Betweenness Centrality", hue="Edge Type", palette=color_dict, kde=True, ax=axs[i, j], legend = False)
                else:
                    axs[i, j].text(0.5, 4, 'N/A', ha='center', va='bottom', fontsize=18)
                    print(f"No data for {cell_title} at threshold {te_thresh}, skipping plot.")
                    continue
                axs[i, j].set_ylabel('')
                axs[i, j].set_xlabel('')
                #if i != 4:
                    #axs[i, j].set_xticklabels([])

        for i, ax in enumerate([axs[0, 0], axs[0, 2], axs[2, 0], axs[2, 2]]):
            ax.annotate(quadrant_titles[i], xy=(1, 1), xytext=(30, 18),
                        xycoords='axes fraction', textcoords='offset points',
                        ha='center', va='baseline', weight='bold', fontsize = 14)

        big_ax.set_xlabel("Betweenness Centrality Values", labelpad=30, fontsize = 14)
        big_ax.set_ylabel("Count", labelpad=40, fontsize = 14)

        # Create custom legend
        legend_elements = [plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=15),
                           plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=15),
                           plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=15)]
        big_ax.legend(legend_elements, [f'Threshold: {te_thresh}', f'Data: {data_name}', 'Plot Type: Betweenness Centrality Histogram'], 
                      loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5, fontsize = 15)

        fig_name = f'hist_betweenness_centrality_quadrants_{te_thresh}.png'
        plt.savefig(f'./aggregates/{fig_name}')

