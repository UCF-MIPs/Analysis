import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import os

def plot_line_violins(graph_dict, data_name):
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
        fig, axs = plt.subplots(4, 1, figsize=(8, 14), sharey=True) # create a 4x1 grid of Axes
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.07, top=0.96, wspace=0.2, hspace=0.3)

        for i in range(4):
            df = pd.DataFrame()  # initialize dataframe
            for cell_title in quadrant_cells[i]:
                graph = graph_dict[cell_title].get(te_thresh, nx.DiGraph())
                degrees = [d for n, d in graph.out_degree()]
                df = df.append(pd.DataFrame({"Degree": degrees, "Edge Type": [cell_title]*len(degrees)}))

            if not df.empty:
                #axs[i].grid(False)
                axs[i].grid(True, alpha=0.5)
                sns.violinplot(x="Edge Type", y="Degree", data=df, palette=color_dict, dodge=False, ax=axs[i])
            else:
                print(f"No data for {quadrant_titles[i]} at threshold {te_thresh}, skipping plot.")
                axs[i].set_xticks(range(len(quadrant_cells[i])))
                axs[i].set_xticklabels(quadrant_cells[i])
                continue
            
            axs[i].set_title(quadrant_titles[i], fontsize = 18, fontweight='bold')
            axs[i].set_ylabel('')
            axs[i].set_xlabel('')
            axs[i].set_xticks(range(len(quadrant_cells[i])))  # Add x-ticks for each cell title in the quadrant
            axs[i].set_xticklabels(quadrant_cells[i])  # Set the labels of the x-ticks to be the cell titles
            
        axs[3].set_xlabel('Network Type', fontsize = 16)
        fig.text(0.01, 0.5, 'Out Degrees', va='center', rotation='vertical', fontsize=16)
   

        # Create custom legend
        legend_elements = [plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=15),
                           plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=15),
                           plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=15)]
        axs[3].legend(legend_elements, [f'Threshold: {te_thresh}', f'Data: {data_name}', 'Plot Type: Out-Degree Violin Plot'],
                      loc='lower center', bbox_to_anchor=(0.47, -0.4), fancybox=True, shadow=True, ncol=5, fontsize = 13)

        #plt.tight_layout()
        fig_name = f'out_degree_thresh_{te_thresh}.png'
        plt.savefig(f'./aggregates/{fig_name}')
        #plt.show()


