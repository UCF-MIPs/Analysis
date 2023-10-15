# Modeling Influence Pathways (MIPs) analysis

## Usage

For basic usage, run the main file.  

```bash

python main.py

```

## Options

Within the main file, functional and plotting options can be adjusted.

To generate hierarchical acyclic tree plots for every possible root node:


```python

generate_trees.generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes=None)

```

