#TODO
# move generated dfs from node_correlation to results folder
# Get participation coeff working
# get ukr/skrip working simultaneously in node_correlation
# put plotting functions into loops
# put participation coeff calculations in loops
# move part coeff function to src

import networkx as nx
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

edge_types = ['TM_*', 'TF_*', 'UM_*', 'UF_*']
dataset = 'ukr_v3' # options: skrip_v4, skrip_v7, ukr_v3

if(dataset=='ukr_v3'):
    name = 'Ukraine'
elif(dataset=='skrip_v7'):
    name = 'Skripal'

in_infl_weights_df = pd.read_csv(f'{dataset}_in_infl_weights_df.csv')
out_infl_weights_df = pd.read_csv(f'{dataset}_out_infl_weights_df.csv')


### COMPARING ALL AGGREGATES

#in_aggr_infl_weights_df = in_infl_weights_df[['actors', 'UM_*', 'UF_*', 'TM_*', 'TF_*']]
in_aggr_infl_weights_df = in_infl_weights_df[['UM_*', 'UF_*', 'TM_*', 'TF_*']]

#out_aggr_infl_weights_df = out_infl_weights_df[['actors', 'UM_*', 'UF_*', 'TM_*', 'TF_*']]
out_aggr_infl_weights_df = out_infl_weights_df[['UM_*', 'UF_*', 'TM_*', 'TF_*']]

# To include plot of corresponding index of in to out plot and out to in plot:
#https://stackoverflow.com/questions/58423707/pandas-matching-index-of-one-dataframe-to-the-column-of-other-dataframe
# Would show that outgoing actors are/are not the same as incoming

# sort by value of first column
out_aggr_infl_weights_df = out_aggr_infl_weights_df.sort_values(by=['UM_*'], ascending=False)
in_aggr_infl_weights_df = in_aggr_infl_weights_df.sort_values(by=['UM_*'], ascending=False)

#inout_df = in_aggr_infl_weights_df['actors']
#pd.merge(inout_df[['actors']], out_aggr_infl_weights_df, how='left', on='actors', sort=False)
#print(inout_df)
#outin_df = in_infl_weights_df[edge_types].reindex(out_aggr_infl_weights_df.index)

# incoming edges plot
heatmap = np.empty((4, len(in_aggr_infl_weights_df['UM_*'])))

for i, (column_name, column_data) in enumerate(in_aggr_infl_weights_df.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data
        pass

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(4))
ax.set_yticklabels(['UM_*', 'UF_*', 'TM_*', 'TF_*'])
ax.set_xlabel('actors')
ax.set_title(f'{dataset} target actor incoming activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_aggr_in_node_activity.png')

'''
# corresponding order for out
heatmap = np.empty((4, 2001))

for i, (column_name, column_data) in enumerate(inout_df.items()):
    print(column_name)
    print(column_data)

    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(4))
ax.set_yticklabels(['UM_*', 'UF_*', 'TM_*', 'TF_*'])
ax.set_xlabel('actors')
ax.set_title('target actor incoming activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig('in2out_node_activity.png')
'''

# outgoing edges plot
heatmap = np.empty((4, len(out_aggr_infl_weights_df['UM_*'])))

for i, (column_name, column_data) in enumerate(out_aggr_infl_weights_df.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(4))
ax.set_yticklabels(['UM_*', 'UF_*', 'TM_*', 'TF_*'])
ax.set_xlabel('actors')
ax.set_title(f'{dataset} source actor outgoing activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_aggr_out_node_activity.png')

'''
# corresponding order for in 
heatmap = np.empty((4, 2001))

for i, (column_name, column_data) in enumerate(outin_df.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(4))
ax.set_yticklabels(['UM_*', 'UF_*', 'TM_*', 'TF_*'])
ax.set_xlabel('actors')
ax.set_title('source actor outgoing activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig('out2in_node_activity.png')
'''





### FOR EACH INFL TYPE

######## UM #########
UM_in_aggr_wdf = in_infl_weights_df[['UM_*', 'UM_UM', 'UM_UF', 'UM_TM', 'UM_TF']]
UM_out_aggr_wdf = out_infl_weights_df[['UM_*', 'UM_UM', 'UM_UF', 'UM_TM', 'UM_TF']]
UM_in_aggr_wdf = UM_in_aggr_wdf.loc[UM_in_aggr_wdf['UM_*'] !=0]
UM_out_aggr_wdf = UM_out_aggr_wdf.loc[UM_out_aggr_wdf['UM_*'] !=0]

#sort
UM_out_aggr_wdf = UM_out_aggr_wdf.sort_values(by=['UM_*'], ascending=False)
UM_in_aggr_wdf = UM_in_aggr_wdf.sort_values(by=['UM_*'], ascending=False)

######## UM out #########
heatmap = np.empty((5, len(UM_out_aggr_wdf['UM_*'])))

for i, (column_name, column_data) in enumerate(UM_out_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['UM_(UM,UF,TM,TF)', 'UM_UM', 'UM_UF', 'UM_TM', 'UM_TF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} UM source actor outgoing activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_UM_out_node_activity.png')
plt.clf()

######## UM in #########
heatmap = np.empty((5, len(UM_in_aggr_wdf['UM_*'])))

for i, (column_name, column_data) in enumerate(UM_in_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['UM_(UM,UF,TM,TF)', 'UM_UM', 'UM_UF', 'UM_TM', 'UM_TF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} UM source actor incoming activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_UM_in_node_activity.png')
plt.clf()




######## TM #########
TM_in_aggr_wdf = in_infl_weights_df[['TM_*', 'TM_TM', 'TM_TF', 'TM_UM', 'TM_UF']]
TM_out_aggr_wdf = out_infl_weights_df[['TM_*', 'TM_TM', 'TM_TF', 'TM_UM', 'TM_UF']]
TM_in_aggr_wdf = TM_in_aggr_wdf.loc[TM_in_aggr_wdf['TM_*'] !=0]
TM_out_aggr_wdf = TM_out_aggr_wdf.loc[TM_out_aggr_wdf['TM_*'] !=0]

TM_out_aggr_wdf = TM_out_aggr_wdf.sort_values(by=['TM_*'], ascending=False)
TM_in_aggr_wdf = TM_in_aggr_wdf.sort_values(by=['TM_*'], ascending=False)

######## TM out #########
heatmap = np.empty((5, len(TM_out_aggr_wdf['TM_*'])))

for i, (column_name, column_data) in enumerate(TM_out_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['TM_(TM,TF,UM,UF)', 'TM_TM', 'TM_TF', 'TM_UM', 'TM_UF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} TM source actor outgoing activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_TM_out_node_activity.png')

######## TM in #########
heatmap = np.empty((5, len(TM_in_aggr_wdf['TM_*'])))

for i, (column_name, column_data) in enumerate(TM_in_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['TM_(TM,TF,UM,UF)', 'TM_TM', 'TM_TF', 'TM_UM', 'TM_UF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} TM source actor incoming activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_TM_in_node_activity.png')






######## UF #########
UF_in_aggr_wdf = in_infl_weights_df[['UF_*', 'UF_UF', 'UF_UM', 'UF_TM', 'UF_TF']]
UF_out_aggr_wdf = out_infl_weights_df[['UF_*', 'UF_UF', 'UF_UM', 'UF_TM', 'UF_TF']]
UF_in_aggr_wdf = UF_in_aggr_wdf.loc[UF_in_aggr_wdf['UF_*'] !=0]
UF_out_aggr_wdf = UF_out_aggr_wdf.loc[UF_out_aggr_wdf['UF_*'] !=0]

UF_out_aggr_wdf = UF_out_aggr_wdf.sort_values(by=['UF_*'], ascending=False)
UF_in_aggr_wdf = UF_in_aggr_wdf.sort_values(by=['UF_*'], ascending=False)

######## UF out #########
heatmap = np.empty((5, len(UF_out_aggr_wdf['UF_*'])))

for i, (column_name, column_data) in enumerate(UF_out_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['UF_(UF,UM,TM,TF)', 'UF_UF', 'UF_UM', 'UF_TM', 'UF_TF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} UF source actor outgoing activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_UF_out_node_activity.png')

######## UF in #########
heatmap = np.empty((5, len(UF_in_aggr_wdf['UF_*'])))

for i, (column_name, column_data) in enumerate(UF_in_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['UF_(UF,UM,TM,TF)', 'UF_UF', 'UF_UM', 'UF_TM', 'UF_TF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} UF source actor incoming activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_UF_in_node_activity.png')





######## TF #########
TF_in_aggr_wdf = in_infl_weights_df[['TF_*', 'TF_TF', 'TF_TM', 'TF_UM', 'TF_UF']]
TF_out_aggr_wdf = out_infl_weights_df[['TF_*', 'TF_TF', 'TF_TM', 'TF_UM', 'TF_UF']]
TF_in_aggr_wdf = TF_in_aggr_wdf.loc[TF_in_aggr_wdf['TF_*'] !=0]
TF_out_aggr_wdf = TF_out_aggr_wdf.loc[TF_out_aggr_wdf['TF_*'] !=0]

TF_out_aggr_wdf = TF_out_aggr_wdf.sort_values(by=['TF_*'], ascending=False)
TF_in_aggr_wdf = TF_in_aggr_wdf.sort_values(by=['TF_*'], ascending=False)

######## TF out #########
heatmap = np.empty((5, len(TF_out_aggr_wdf['TF_*'])))

for i, (column_name, column_data) in enumerate(TF_out_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['TF_(TF,TM,UM,UF)', 'TF_TF', 'TF_TM', 'TF_UM', 'TF_UF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} TF source actor outgoing activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_TF_out_node_activity.png')

######## TF in #########
heatmap = np.empty((5, len(TF_in_aggr_wdf['TF_*'])))

for i, (column_name, column_data) in enumerate(TF_in_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['TF_(TF,TM,UM,UF)', 'TF_TF', 'TF_TM', 'TF_UM', 'TF_UF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} TF source actor incoming activity')

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_TF_in_node_activity.png')



# participation coefficient

def part_coeff(df, sour_infl):
    # probably should include actors and go row by row
    m=4
    types = ['UF', 'UM', 'TF', 'TM']
    aggr_type = str(sour_infl + '_*')
    o = df[aggr_type].to_numpy()
    ptemp = np.zeros_like(o)
    for i in types:
        infl_type = str(sour_infl + '_' + i)
        #print(infl_type)
        #print(k.shape)
        #print(np.count_nonzero(k))
        k = df[infl_type].to_numpy()
        #print(np.count_nonzero(o))
        ptemp += np.divide(k,o,out=np.zeros_like(k), where=o!=0, dtype=float)**2
    p = (4./3.)*ptemp
    return p # array of part. coeffs


part_cof_UM = part_coeff(out_infl_weights_df, 'UM')
part_cof_TM = part_coeff(out_infl_weights_df, 'TM')
part_cof_UF = part_coeff(out_infl_weights_df, 'UF')
part_cof_TF = part_coeff(out_infl_weights_df, 'TF')


part_df = pd.DataFrame({'actors': out_infl_weights_df['actors'],\
                        'p_UM': part_cof_UM,\
                        'p_TM': part_cof_TM,\
                        'p_UF': part_cof_UF,\
                        'p_TF': part_cof_TF})


part_df.to_csv(f'{dataset}_part_coef.csv')










