import networkx as nx
import pandas as pd
import numpy as np
from src import (plot_quadrant_bc, plot_quadrant_violins, compute_basic_metrics, generate_edge_types, plot_BC_hist_aggr,
                 plot_network_degree_centrality, plot_violin_degrees_by_threshold, plot_total_nodes_edges_networks, 
                 plot_network_BC, plot_violin_out_degree, plot_total_nodes_networks, plot_total_edges_networks, 
                 plot_node_BC_dist, plot_edge_BC_dist, plot_quadrant_violins, plot_line_violins, plot_line_hist, plot_violin_BC,
                 plot_box_BC, get_top_n_by_bc, plot_line_data_alex)


class NetworkAnalysis:
    def __init__(self, data_name, thresholds):
        self.data_name = data_name
        self.thresholds = thresholds
        self.cascade_df = None
        self.actor_df = None
        self.actors = None
        self.edge_types = None
        self.graph_dict = None
        self.all_degrees_dict = None

    def load_data(self):
        self.edge_types = generate_edge_types.generate_edge_types()
        if self.data_name == 'skripal':
            self.cascade_df = pd.read_csv(f'data/{self.data_name}/actor_te_edges_2018_03_01_2018_05_01.csv'.format(self.data_name))
            self.cascade_df = self.cascade_df.loc[(self.cascade_df['Source'] > 1.) & (self.cascade_df['Target'] > 1.) & (self.cascade_df['Source']<101.) & (self.cascade_df['Target']<101.)]

            self.actor_df = pd.read_csv(f'data/{self.data_name}/actors_v4.csv')
            self.actors = dict(zip(self.actor_df.actor_id, self.actor_df.actor_label))
        else:
            self.cascade_df = pd.read_csv(f'data/{self.data_name}/Actor_TE_Edges_Ukraine_v1.csv'.format(self.data_name))
            self.cascade_df = self.cascade_df.rename(columns={'all_all': 'total_te'})
            self.cascade_df = self.cascade_df[['Source', 'Target', 'TF_TF', 'TF_TM', 'TF_UF', 'TF_UM',
                'TM_TF', 'TM_TM', 'TM_UF', 'TM_UM', 'UF_TF', 'UF_TM', 'UF_UF', 'UF_UM',
                'UM_TF', 'UM_TM', 'UM_UF', 'UM_UM', 'total_te']]
            self.actor_df = pd.read_csv(f'data/{self.data_name}/actors_Ukraine_v1.csv')
            self.actors = dict(zip(self.actor_df.actor_id, self.actor_df.actor_label))

    def generate_graphs(self):
        self.graph_dict = {}
        for edge_type in self.edge_types:
            self.graph_dict[edge_type] = {}
            for te_thresh in self.thresholds:
                graph_df = self.cascade_df.loc[(self.cascade_df[edge_type] > te_thresh)]
                g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                self.graph_dict[edge_type][te_thresh] = g
        return self.graph_dict

    #def generate_graphs1(self, te_thresholds):
    #    self.graph_dict = {}
    #    for edge_type in self.edge_types:
    #        self.graph_dict[edge_type] = {}
    #        for te_thresh in te_thresholds:
    #            graph_df = self.cascade_df.loc[(self.cascade_df[edge_type] > te_thresh)]
    #            g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
    #            self.graph_dict[edge_type][te_thresh] = g
    #    return self.graph_dict

    #def calculate_metrics(self, graph_dict):
    #    metrics_dict = {}
    #    
    #    for edge_type, threshold_graphs in graph_dict.items():
    #        metrics_dict[edge_type] = {}
    #        
    #        for te_threshold, graph in threshold_graphs.items():
    #            # Calculate the betweenness centrality values
    #            bc_values = nx.betweenness_centrality(graph, normalized=False)
#
    #            # Calculate the total out-degree
    #            out_degree_sum = sum([out_degree for node, out_degree in graph.out_degree()])
#
    #            # Store the metrics in a dictionary
    #            metrics = {
    #                #'betweenness_centrality_values': bc_values,
    #                'total_out_degree': out_degree_sum,
    #            }
#
    #            # Save the metrics in the overall metrics dictionary
    #            metrics_dict[edge_type][te_threshold] = metrics
    #    return metrics_dict


    def compute_all_metrics(self, graph_dict):
        metrics = compute_basic_metrics.compute_basic_metrics(graph_dict)
        top_nodes_bc = get_top_n_by_bc.get_top_n_by_bc(graph_dict, self.actors, 0.2)
        return top_nodes_bc


    #def generate_plots(self):

        #total_out_degrees = 

        #plot_line_data_alex.plot_line_data_alex()

        ########### Bar plots of total edges and nodes ########### 

        #plot_total_nodes_edges_networks.plot_total_nodes_edges_networks(self.graph_dict, self.data_name)
        #plot_violin_out_degree.plot_violin_out_degree(self.graph_dict)
        #plot_total_nodes_networks.plot_total_nodes_networks(self.graph_dict, self.data_name)
        #plot_total_edges_networks.plot_total_edges_networks(self.graph_dict, self.data_name)
        #plot_violin_BC.plot_violin_BC(self.graph_dict)
        #plot_box_BC.plot_box_BC(self.graph_dict, self.data_name)

        ########### Individual histograms of nodes and edges distributions ########### 

        #plot_node_BC_dist.plot_node_BC_dist(self.graph_dict)
        #plot_edge_BC_dist.plot_edge_BC_dist(self.graph_dict)

        ########### Network graphs ########### 

        #plot_network_BC.plot_network_BC(self.graph_dict, self.actors, self.data_name)
        #plot_network_degree_centrality.plot_network_degree_centrality(self.graph_dict, self.actors, self.data_name)

        ########### Aggregates ###########

        #plot_line_violins.plot_line_violins(self.graph_dict, self.data_name)
        #plot_line_hist.plot_line_hist(self.graph_dict, self.data_name)
        #plot_BC_hist_aggr.plot_BC_hist_aggr(self.graph_dict)
        #plot_quadrant_violins.plot_quadrant_violins(self.graph_dict, self.data_name)
        #plot_quadrant_bc.plot_quadrant_bc(self.graph_dict, self.data_name)
        #plot_violin_out_degree.plot_violin_out_degree(self.graph_dict)

    #def generate_csv(self):


def main():
    net_analysis = NetworkAnalysis(data_name='skripal', thresholds = [0.05, 0.1, 0.15])
    net_analysis.load_data()
    graph_dict = net_analysis.generate_graphs()
    top_nodes_bc = net_analysis.compute_all_metrics(graph_dict)
    print('---------------------', top_nodes_bc['TM_UF'][0.1])
    #te_thresholds = np.arange(0, 1.01, 0.01)
    #graphs_dict = net_analysis.generate_graphs1(te_thresholds)
    #res1 = net_analysis.calculate_metrics(graphs_dict)
    #plot_line_data_alex.plot_line_data_alex(res1, 'skripal')
    
    #net_analysis.generate_plots()

if __name__ == '__main__':
    main()
