import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.offsetbox import AnchoredText
import matplotlib.lines as mlin



def plot_htrees(xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos, te_thresh, edge_type):
    '''
    horizontal trees/hierarchical directed graph propogation
    input:
    ...
    path: strongest pathway selection method: None, greedy, or summed (total edge weight)
    '''
    figs = []
    for tree, pathway, colormap_nodes, colormap_edges, pos in zip (xtrees, xpathways, xcolormap_nodes, xcolormap_edges, xpos):
        fig, ax = plt.subplots(figsize=(7,16))
        nx.draw(tree, pos, node_color=colormap_nodes, edge_color=colormap_edges, with_labels=True, width=3, font_size=12, node_size=250, ax = ax)
        #short
        #fig = plt.figure(3,figsize=(20,20))
        #tall
        node_type = ['Expanded', 'Terminal', 'Unexpanded']
        te_text = str('TE threshold: ' + str(te_thresh))
        text_box = AnchoredText(te_text, frameon=True, loc='lower left', pad=0.5)
        #ax.setp(text_box.patch, facecolor='white', alpha=0.5)
        ax.add_artist(text_box)
        te_text2 = str('Influence type: ' + '\n' + str(edge_type))
        text_box2 = AnchoredText(te_text2, frameon=True, loc='lower center', pad=0.5)
        ax.add_artist(text_box2)
        line1 = mlin.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="#1f75ae")
        line2 = mlin.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="green")
        line3 = mlin.Line2D([], [], color="white", marker='o', markersize=15,  markerfacecolor="yellow")
        ax.legend((line1, line2, line3), ('Expanded', 'Terminal', 'Unexpanded'), numpoints=1, loc='lower right')
        figs.append(ax)
    return figs




