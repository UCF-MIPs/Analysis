basic functionality of static analysis for use in MIPs interface

Static analysis input and output

input: 
g = TE network 
s = starting point type (most influential nodes/most influential edges), 
n = number of starting points
t = pathway selection method (none/greedy/summed path)

output: 
tree for each influence type in form of the following table:
source, target, target count (how many times is the node repeated)
dictionary of n pathways per tree 
ex// {UF-TM: (edgelist1, edgelist2)}
tree has embedded auto-threshold, label the value somewhere
