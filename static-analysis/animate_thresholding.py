import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.offsetbox import AnchoredText
from string import ascii_lowercase as letters

# Directories
thresh_anim = 'results/thresh_anim/'

# Parameters
max_nodes = 101.
te_threshs=np.round(np.arange(0.0,1.0,0.05), decimals=2).tolist()
print(te_threshs)

for te_thresh, l in zip(te_threshs, letters):
    graph_df = pd.read_csv('data/dynamic/actor_te_edges_2018_03_01_2018_05_01.csv')
    cascade_df = graph_df.loc[(graph_df['total_te'] > te_thresh) & (graph_df['Target'] > 1.) \
            & (graph_df['Source']<max_nodes) & (graph_df['Target']<max_nodes)]

    actor_df = pd.read_csv('data/dynamic/actors.csv')
    actors = dict(zip(actor_df.actor_id, actor_df.actor_label))

    g = nx.from_pandas_edgelist(cascade_df, 'Source', 'Target', 'total_te', create_using=nx.DiGraph())

    #nx.draw(g, pos = nx.random_layout(g), node_size=20, width=0.1)
    nx.draw(g, pos = nx.circular_layout(g), node_size=20, width=0.1)
    
    te_text = f"TE_threshold: {te_thresh}"
    text_box = AnchoredText(te_text, frameon=True, loc='lower left', pad=0.5, prop=dict(size=14.0))
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box)

    filename = f"{thresh_anim}/{l}_draw_{te_thresh}.png"

    plt.savefig(filename)
    plt.clf()

