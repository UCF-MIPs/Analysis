import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.offsetbox import AnchoredText
import matplotlib.lines as mlin
import numpy as np
from scipy import signal

def plot_all_sweep(te_threshes, edge_types, metric, csv_name):
    '''
    metric sweeps over TE threshold
    x: list - same across all influence types (threshold for instance)
    df: dict - unique to each influence type
    edge_type: str - for naming plot
    metric: str - y axis metric
    '''
    y = {}
    if(metric=='outdegree'):
        maxval = 0
        xmax = 0
        for edge_type in edge_types:
            out_degree = []
            for te_thresh in te_threshes:
                cascade_df = pd.read_csv(csv_name, usecols=['Source', 'Target', edge_type])
                graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
                g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                #nx.relabel_nodes(g, actors, copy=False)
                deg = sum(dict(g.out_degree()).values())
                out_degree.append(deg)
                if(deg > maxval):
                    maxval=deg
                if(deg is not 0 and te_thresh > xmax):
                    xmax = te_thresh
            y[edge_type] = out_degree
        for edge_type in edge_types:
            # 16 plots ...
            fig, ax = plt.subplots()
            yn = np.asarray(y[edge_type])
            #yn_smooth = signal.savgol_filter(yn, 11, 3)
            ax.plot(te_threshes, yn)
            plt.xlabel('TE thresh')
            plt.ylabel('sum of out degree')
            plt.ylim(0,maxval)
            plt.xlim(0, xmax)
            plt.savefig(f'{edge_type}_{metric}_thresh_sweep.png')
    

    if(metric=='bc'):
        maxval = 0
        xmax = 0
        for edge_type in edge_types:
            bc = []
            for te_thresh in te_threshes:
                cascade_df = pd.read_csv(csv_name, usecols=['Source', 'Target', edge_type])
                graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
                g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                #nx.relabel_nodes(g, actors, copy=False)
                bc_temp = sum(dict(nx.betweenness_centrality(g)).values())
                bc.append(bc_temp)
                if(bc_temp > maxval):
                    maxval = bc_temp
                if(bc_temp is not 0 and te_thresh > xmax):
                    xmax = te_thresh
            y[edge_type] = bc
        for edge_type in edge_types:
            fig, ax = plt.subplots()
            yn = np.asarray(y[edge_type])
            #yn_smooth = signal.savgol_filter(yn, 3, 5)
            ax.plot(te_threshes, yn)
            plt.xlabel('TE thresh')
            plt.ylabel('sum of betweenness centrality')
            plt.ylim(0, maxval)
            plt.xlim(0,xmax)
            plt.savefig(f'{edge_type}_{metric}_thresh_sweep.png')
    

    if(metric=='num_nodes'):
        maxval = 0
        xmax = 0
        for edge_type in edge_types:
            num_nodes = []
            for te_thresh in te_threshes:
                cascade_df = pd.read_csv(csv_name, usecols=['Source', 'Target', edge_type])
                graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
                g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                #nx.relabel_nodes(g, actors, copy=False)
                num_nodes_temp = g.number_of_nodes()
                num_nodes.append(num_nodes_temp)
                if(num_nodes_temp > maxval):
                    maxval = num_nodes_temp
                if(num_nodes_temp is not 0 and te_thresh > xmax):
                    xmax = te_thresh
            y[edge_type] = num_nodes
        for edge_type in edge_types:
            fig, ax = plt.subplots()
            yn = np.asarray(y[edge_type])
            #yn_smooth = signal.savgol_filter(yn, 3, 5)
            ax.plot(te_threshes, yn)
            plt.xlabel('TE thresh')
            plt.ylabel('number of nodes')
            plt.ylim(0, maxval)
            plt.xlim(0, xmax)
            plt.savefig(f'{edge_type}_{metric}_thresh_sweep.png')


    if(metric=='num_edges'):
        maxval = 0
        xmax = 0
        for edge_type in edge_types:
            num_edges = []
            for te_thresh in te_threshes:
                cascade_df = pd.read_csv(csv_name, usecols=['Source', 'Target', edge_type])
                graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
                g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                #nx.relabel_nodes(g, actors, copy=False)
                num_edges_temp = g.number_of_edges()
                num_edges.append(num_edges_temp)
                if(num_edges_temp > maxval):
                    maxval = num_edges_temp
                if(num_edges_temp is not 0 and te_thresh > xmax):
                    xmax = te_thresh
            y[edge_type] = num_edges
        for edge_type in edge_types:
            fig, ax = plt.subplots()
            yn = np.asarray(y[edge_type])
            #yn_smooth = signal.savgol_filter(yn, 3, 5)
            ax.plot(te_threshes, yn)
            plt.xlabel('TE thresh')
            plt.ylabel('number of edges')
            plt.ylim(0, maxval)
            plt.xlim(0, xmax)
            plt.savefig(f'{edge_type}_{metric}_thresh_sweep.png')




def plot_sweep(te_threshes, cascade_df, edge_type, metric):
    '''
    metric sweeps over TE threshold
    x: list - same across all influence types (threshold for instance)
    cascade_df:
    edge_type: str - for naming plot
    metric: str - y axis metric
    '''
    if(metric=='outdegree'):
        out_degree = []
        for te_thresh in te_threshes:
            graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
            g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
            #nx.relabel_nodes(g, actors, copy=False)
            out_degree.append(len(g.out_degree()))

        y = out_degree
        fig, ax = plt.subplots()
        yn = np.asarray(y)
        #yn_fit = np.polyfit(x, yn, 3)
        yn_smooth = signal.savgol_filter(yn, 11, 3)
        ax.plot(yn_smooth)
        #ax.plot(yn_fit)
        plt.xlabel('TE thresh')
        plt.ylabel('out degree')
        plt.savefig(f'{edge_type}_{metric}_thresh_sweep.png')
    

    if(metric=='bc'):
        bc = []
        for te_thresh in te_threshes:
            graph_df = cascade_df.loc[(cascade_df[edge_type] > te_thresh)]
            g = nx.from_pandas_edgelist(graph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
            nx.relabel_nodes(g, actors, copy=False)
            bc.append(nx.betweenness_centrality(g))

        y = bc
        fig, ax = plt.subplots()
        yn = np.asarray(y)
        #yn_fit = np.polyfit(x, yn, 3)
        yn_smooth = signal.savgol_filter(yn, 3, 5)
        #ax.plot(yn_fit)
        print(yn_smooth)
        plt.xlabel('TE thresh')
        ply.ylabel('out degree')
        plt.savefig(f'{edge_type}_{metric}_thresh_sweep.png')
    

