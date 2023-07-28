import networkx as nx
import pandas as pd
from src import (plot_quadrant_bc, plot_quadrant_violins,
                 generate_edge_types, compute_degrees_nodes_edges, plot_BC_hist_aggr,
                 plot_network_degree_centrality, plot_violin_degrees_by_threshold, plot_total_nodes_edges_networks, 
                 plot_network_BC, plot_violin_out_degree, plot_total_nodes_networks, plot_total_edges_networks, 
                 plot_node_BC_dist, plot_edge_BC_dist, plot_quadrant_violins, plot_line_violins, plot_line_hist)


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
        if self.data_name == 'skripal' or self.data_name == 'Skripal':
            self.cascade_df = pd.read_csv(f'data/{self.data_name}/actor_te_edges_2018_03_01_2018_05_01.csv'.format(self.data_name))
            self.cascade_df = self.cascade_df.loc[(self.cascade_df['Source'] > 1.) & (self.cascade_df['Target'] > 1.) & (self.cascade_df['Source']<101.) & (self.cascade_df['Target']<101.)]

            self.actor_df = pd.read_csv(f'data/{self.data_name}/actors_v4.csv')
            self.actors = dict(zip(self.actor_df.actor_id, self.actor_df.actor_label))

            self.edge_types = generate_edge_types.generate_edge_types()

        else:
            self.cascade_df = pd.read_csv(f'data/{self.data_name}/Actor_TE_Edges_Ukraine_v1.csv'.format(self.data_name))
            self.cascade_df = self.cascade_df.loc[(self.cascade_df['Source'] > 1.) & (self.cascade_df['Target'] > 1.) & (self.cascade_df['Source']<101.) & (self.cascade_df['Target']<101.)]

            self.actor_df = pd.read_csv(f'data/{self.data_name}/actors_Ukraine_v1.csv')
            self.actors = dict(zip(self.actor_df.actor_id, self.actor_df.actor_label))

            self.edge_types = generate_edge_types.generate_edge_types()


    def generate_graphs(self):
        self.graph_dict = {}
        for edge_type in self.edge_types:
            self.graph_dict[edge_type] = {}
            for te_thresh in self.thresholds:
                graph_df = self.cascade_df.loc[(self.cascade_df[edge_type] > te_thresh)]
                g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                self.graph_dict[edge_type][te_thresh] = g

    def compute_all_degrees(self):
        self.all_degrees_dict = {}
        for edge_type, threshold_dict in self.graph_dict.items():
            self.all_degrees_dict[edge_type] = {}
            for threshold, graph in threshold_dict.items():
                self.all_degrees_dict[edge_type][threshold] = compute_degrees_nodes_edges.compute_degrees_nodes_edges(graph)

    def generate_plots(self):

        ########### Bar plots of total edges and nodes ########### 

        #plot_total_nodes_edges_networks.plot_total_nodes_edges_networks(self.graph_dict, self.data_name)
        #plot_total_nodes_networks.plot_total_nodes_networks(self.graph_dict, self.data_name)
        #plot_total_edges_networks.plot_total_edges_networks(self.graph_dict, self.data_name)

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
        plot_quadrant_violins.plot_quadrant_violins(self.graph_dict, self.data_name)
        plot_quadrant_bc.plot_quadrant_bc(self.graph_dict, self.data_name)

        #plot_violin_out_degree.plot_violin_out_degree(self.graph_dict)

    #def generate_csv(self):


def main():
    net_analysis = NetworkAnalysis(data_name='skripal', thresholds=[0.01, 0.1, 0.15, 0.2])
    net_analysis.load_data()
    net_analysis.generate_graphs()
    net_analysis.generate_plots()

if __name__ == '__main__':
    main()
