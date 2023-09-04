import networkx as nx
from . import strongest_path_greedy, strongest_path_summed
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque
import matplotlib.pyplot as plt

def generate_tree_plots(g, edge_type, te_thresh, pathway_selection, root_nodes, dir_name): # derived from modification of Alex Baekey's generate_tree_plots method and htrees method
	# if root nodes are not provided, generate a some using a default method
	if root_nodes is None:
		root_nodes = [x for x in g.nodes() if g.out_degree(x) > 0]

	# Generate lists containing trees and information about trees
	rnodes = root_nodes
	xtrees = make_trees(g, rnodes)
	xcolormap_nodes = make_node_colormaps(g, xtrees)
	xpathways, xstrengths = make_pathways(pathway_selection, xtrees, g, rnodes)
	xcolormap_edges = make_edge_colormaps(xtrees, xpathways)
	xpos = make_poses(rnodes, xtrees)
	xlabel_nodes = make_node_labels(xtrees)
	make_figs(rnodes, xtrees, xcolormap_nodes, xpathways, xstrengths, xcolormap_edges, xpos, xlabel_nodes, edge_type, te_thresh, dir_name)

def make_trees(g, roots):
	# make a list of tree graphs by calling make_tree on each root individually
	xtrees = []
	for root in roots:
		xtrees.append(make_tree(g, root))
	return xtrees

def make_tree(g, source): # derived from modification of NetworkX's bfs_tree method
	# Given an networkX graph and a node name to use as the root node, construct a tree of all nodes reachable from the root, allowing duplicates
	T = nx.DiGraph() # initialize empty graph to represent the tree
	T.add_node(source) # add the root node
	edges_gen = my_edge_bfs(g, source) # perform edge_bfs on g to obtain edges and their associated nodes reachable from source
	T.add_edges_from(edges_gen) # add the edges obtained from edges_gen to the tree
	return T

def my_edge_bfs (g, source): # derived from modification of networkX's edge_bfs method
	# initialize the list of nodes
	nodes = list(g.nbunch_iter(source)) 
	# define a lambda function to lookup a node's outgoing edges
	edges_from = lambda node: iter(g.edges(node)) 
	# initialize the set of visited nodes and edges
	visited_nodes = set()
	visited_edges = set()
	# variable tracking the number of spaces to append to node names to guarantee unique node aliases, increments by 1 each time it is used 
	alias_repetitions = 1
	# initialize queue of nodes and outgoing edges edges to visit
	queue = deque([(n, edges_from(n)) for n in nodes]) 
	# while the queue is not empty, explore the unvisited edges of the graph
	while queue: 
		parent, children_edges = queue.popleft() 
		visited_nodes.add(parent)
		for edge in children_edges: 
			if edge not in visited_edges:
				child = edge[1]
				# If the edge and child node have not been explored, add its child node and the child node's outgoing edges to the queue, and yield the current edge
				if child not in visited_nodes:
					visited_nodes.add(child)
					visited_edges.add(edge)
					queue.append((child, edges_from(child)))
					yield edge
				# If the edge has not been explored but the child node has been, yield an edge with an alias for the child's name, so that it will appear as a distinct node during tree construction
				else:
					visited_edges.add(edge)
					child_alias = edge[1]+(' '*alias_repetitions)
					edge_alias = (edge[0], child_alias)
					alias_repetitions+=1
					yield edge_alias

def make_node_colormaps(g, xtrees):
	xcolormap_nodes = []
	for t in xtrees:
		xcolormap_nodes.append(make_node_colormap(g,t))
	return xcolormap_nodes

def make_node_colormap(g, t): # derived from modification of Alex Baekey's htrees method
	colormap_nodes = []
	for node in t: # for each node in the tree, color it green if it is not a leaf node and #1f78b4 if it is a leaf node
		if g.out_degree(node.rstrip())==0:
			colormap_nodes.append('green')
		else:
			colormap_nodes.append('#1f78b4')
	return colormap_nodes

def make_pathways(path, xtrees, g, rnodes):
	xpaths = []
	xstrengths = []
	if path == None:
		return None
	else:
		for i in range(len(rnodes)):
			p, s = make_pathway(path, xtrees[i], g, rnodes[i])
			xpaths.append(p)
			xstrengths.append(s)
	return xpaths, xstrengths

def make_pathway(path, t, g, r): # derived from modification of Alex Baekey's htrees method
	if path == 'greedy':
		pathway, strength = strongest_path_greedy.strongest_path_greedy(t,g,r)
	elif path == 'summed':
		pathway, strength = strongest_path_summed.strongest_path_summed(t,g,r)
	else:
		raise Exception("Invalid choice of path, must be None, 'greedy', or 'summed'")
	return pathway, strength

def make_edge_colormaps(xtrees, xpaths):
	xcolormap_edges = []
	for i in range(len(xtrees)):
		xcolormap_edges.append(make_edge_colormap(xtrees[i], xpaths[i]))
	return xcolormap_edges

def make_edge_colormap(t, p): # derived from modification of Alex Baekey's htrees method
	colormap_edges = []
	print(p)
	for edge in t.edges:
		if(edge in p):
			colormap_edges.append('red')
		else:
			colormap_edges.append('black')
	return colormap_edges

def make_poses(rnodes, xtrees, min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0):
	xpos = []
	for i in range(len(rnodes)):
		xpos.append(make_pos(rnodes[i], xtrees[i]))
	return xpos

def make_pos(r, t, min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0):
	# dictionary mapping each node name to its coordinates
	pos = {}
	# list mapping out how many layers are in the tree, and how many nodes are in each layer, initialize with root node
	l = [[r]]
	# determine how many nodes are in the next layer
	c = []
	for node in l[-1]:
		for n in t.successors(node):
			c.append(n)

	# add the next layer to l, and determine how many nodes are in the next layer after that, continuing as long as the new layer being explored is not empty
	while c:
		l.append(c)
		c = []
		for node in l[-1]:
			for n in t.successors(node):
				c.append(n)

	# determine how much horizontal space in the figure to allocate to each layer of the tree
	horizontal_share = (max_x - min_x)/(len(l))

	# determine how much horizontal space in the figure to allocate to each node in each layer of the tree
	vertical_share = []
	for layer in l:
		vertical_share.append((max_y - min_y)/(len(layer)+1))

	# assign coordiantes to each node
	curr_x_min = min_x
	curr_y_min = min_y
	for layer in range(len(l)):
		for node in l[layer]:
			pos[node] = (curr_x_min, (curr_y_min + vertical_share[layer])/2)
			curr_y_min += vertical_share[layer]
		curr_y_min = min_y
		curr_x_min += horizontal_share

	return pos

def make_node_labels(xtrees):
	xlabel_nodes = []
	for t in xtrees:
		xlabel_nodes.append(make_node_label(t))
	return xlabel_nodes

def make_node_label(t):
	labels = {}
	for n in list(t):
		labels[n]=n.rstrip() 
	return(labels)

def make_figs(rnodes, xtrees, xcolormap_nodes, xpathways, xstrengths, xcolormap_edges, xpos, xlabel_nodes, edge_type, te_thresh, dir_name): # derived from modification of Alex Baekey's htrees method
	for i in range(len(rnodes)):
		plt.figure(num=i)
		plt.axis("off")
		nx.draw_networkx_nodes(xtrees[i], xpos[i], node_color=xcolormap_nodes[i], node_size=50)
		nx.draw_networkx_edges(xtrees[i], xpos[i], edge_color=xcolormap_edges[i])
		nx.draw_networkx_labels(xtrees[i], xpos[i],labels=xlabel_nodes[i], font_size=8, verticalalignment='top')

		if dir_name is not None:
			plt.savefig(f'{dir_name}/{edge_type}_te_thresh{te_thresh}_root{rnodes[i]}.png')#, bbox_inches='tight')
		else: 
			plt.savefig(f'{edge_type}_te_thresh{te_thresh}_root{rnodes[i]}.png', bbox_inches='tight')

		plt.close(fig=i)

		#plt.show()