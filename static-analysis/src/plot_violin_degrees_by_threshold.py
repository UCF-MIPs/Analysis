import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_violin_degrees_by_threshold(graph_dict):
    
    #color_dict = {'UF': '#1ABC9C', 'UM': '#F1C40F', 'TF': '#E84393', 'TM': '#27AE60', 'to' : '#E74C3C'}
    
    # get all unique thresholds
    thresholds = sorted([thresh for edge_type, threshold_dict in graph_dict.items() for thresh in threshold_dict.keys()])
    thresholds = list(set(thresholds))  # remove duplicates and sort

    # get all unique edge types
    edge_types = list(graph_dict.keys())
    
    # Iterate over each edge_type
    for edge_type in edge_types:
        degree_values = []
        all_thresholds = []
        
        # iterate through each threshold for this edge_type
        for thresh in thresholds:
            if thresh in graph_dict[edge_type]:
                G = graph_dict[edge_type][thresh]
                
                # calculate degree for each node in the current graph
                degrees = [d for n, d in G.out_degree()]
                
                # add degree values and corresponding threshold to lists
                degree_values.extend(degrees)
                all_thresholds.extend([thresh]*len(degrees))
                
        # convert degree_values and all_thresholds to a DataFrame
        df = pd.DataFrame({"Degree": degree_values, "Threshold": all_thresholds})

        plt.figure(figsize=(10, 5))
        plt.style.use("seaborn")
        plt.grid(True, alpha = 0.3)
        
        # create violin plot
        sns.violinplot(x="Threshold", y="Degree", data=df)
        
        plt.title(f'Out Degrees of {edge_type} Network by Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Out Degree Value')
        plt.tight_layout()
        fig_name = f'{edge_type}_violin_degree_by_threshold.png'
        plt.savefig(f'./degree_plots/distributions/{fig_name}')
        plt.close()  # close the figure to free up memory
