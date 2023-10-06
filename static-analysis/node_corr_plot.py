import networkx as nx
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
<<<<<<< HEAD
=======
import matplotlib as mpl
>>>>>>> dev-ab
plt.ion()

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

edge_types = ['TM_*', 'TF_*', 'UM_*', 'UF_*']
dataset = 'ukr_v3' # options: skrip_v4, skrip_v7, ukr_v3

<<<<<<< HEAD
=======
if(dataset=='ukr_v3'):
    name = 'Ukraine'
elif(dataset=='skrip_v7'):
    name = 'Skripal'

>>>>>>> dev-ab
in_infl_weights_df = pd.read_csv(f'{dataset}_in_infl_weights_df.csv')
out_infl_weights_df = pd.read_csv(f'{dataset}_out_infl_weights_df.csv')


<<<<<<< HEAD

=======
>>>>>>> dev-ab
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
<<<<<<< HEAD
ax.set_title('{dataset} source actor outgoing activity')
=======
ax.set_title(f'{dataset} source actor outgoing activity')
>>>>>>> dev-ab

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
<<<<<<< HEAD
UM_in_aggr_wdf = UM_out_aggr_wdf.loc[UM_out_aggr_wdf['UM_*'] !=0]
=======
UM_in_aggr_wdf = UM_in_aggr_wdf.loc[UM_in_aggr_wdf['UM_*'] !=0]
>>>>>>> dev-ab
UM_out_aggr_wdf = UM_out_aggr_wdf.loc[UM_out_aggr_wdf['UM_*'] !=0]

#sort
UM_out_aggr_wdf = UM_out_aggr_wdf.sort_values(by=['UM_*'], ascending=False)
UM_in_aggr_wdf = UM_in_aggr_wdf.sort_values(by=['UM_*'], ascending=False)

<<<<<<< HEAD
# UM outgoing edges plot
heatmap = np.empty((5, len(UM_in_aggr_wdf['UM_*'])))
=======
######## UM out #########
heatmap = np.empty((5, len(UM_out_aggr_wdf['UM_*'])))
>>>>>>> dev-ab

for i, (column_name, column_data) in enumerate(UM_out_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

<<<<<<< HEAD
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(5))
ax.set_yticklabels(['UM_*', 'UM_UM', 'UM_UF', 'UM_TM', 'UM_TF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{dataset} UM source actor outgoing activity')
=======
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['UM_(UM,UF,TM,TF)', 'UM_UM', 'UM_UF', 'UM_TM', 'UM_TF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} UM source actor outgoing activity')
>>>>>>> dev-ab

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_UM_out_node_activity.png')
<<<<<<< HEAD
=======
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
>>>>>>> dev-ab




######## TM #########
TM_in_aggr_wdf = in_infl_weights_df[['TM_*', 'TM_TM', 'TM_TF', 'TM_UM', 'TM_UF']]
TM_out_aggr_wdf = out_infl_weights_df[['TM_*', 'TM_TM', 'TM_TF', 'TM_UM', 'TM_UF']]
<<<<<<< HEAD
TM_in_aggr_wdf = TM_out_aggr_wdf.loc[TM_out_aggr_wdf['TM_*'] !=0]
=======
TM_in_aggr_wdf = TM_in_aggr_wdf.loc[TM_in_aggr_wdf['TM_*'] !=0]
>>>>>>> dev-ab
TM_out_aggr_wdf = TM_out_aggr_wdf.loc[TM_out_aggr_wdf['TM_*'] !=0]

TM_out_aggr_wdf = TM_out_aggr_wdf.sort_values(by=['TM_*'], ascending=False)
TM_in_aggr_wdf = TM_in_aggr_wdf.sort_values(by=['TM_*'], ascending=False)

<<<<<<< HEAD
# TM outgoing edges plot
heatmap = np.empty((5, 2001))
heatmap = np.empty((5, len(TM_in_aggr_wdf['TM_*'])))
=======
######## TM out #########
heatmap = np.empty((5, len(TM_out_aggr_wdf['TM_*'])))
>>>>>>> dev-ab

for i, (column_name, column_data) in enumerate(TM_out_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

<<<<<<< HEAD
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(5))
ax.set_yticklabels(['TM_*', 'TM_TM', 'TM_TF', 'TM_UM', 'TM_UF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{dataset} TM source actor outgoing activity')
=======
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['TM_(TM,TF,UM,UF)', 'TM_TM', 'TM_TF', 'TM_UM', 'TM_UF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} TM source actor outgoing activity')
>>>>>>> dev-ab

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_TM_out_node_activity.png')

<<<<<<< HEAD
=======
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


>>>>>>> dev-ab




######## UF #########
UF_in_aggr_wdf = in_infl_weights_df[['UF_*', 'UF_UF', 'UF_UM', 'UF_TM', 'UF_TF']]
UF_out_aggr_wdf = out_infl_weights_df[['UF_*', 'UF_UF', 'UF_UM', 'UF_TM', 'UF_TF']]
<<<<<<< HEAD
UF_in_aggr_wdf = UF_out_aggr_wdf.loc[UF_out_aggr_wdf['UF_*'] !=0]
=======
UF_in_aggr_wdf = UF_in_aggr_wdf.loc[UF_in_aggr_wdf['UF_*'] !=0]
>>>>>>> dev-ab
UF_out_aggr_wdf = UF_out_aggr_wdf.loc[UF_out_aggr_wdf['UF_*'] !=0]

UF_out_aggr_wdf = UF_out_aggr_wdf.sort_values(by=['UF_*'], ascending=False)
UF_in_aggr_wdf = UF_in_aggr_wdf.sort_values(by=['UF_*'], ascending=False)

<<<<<<< HEAD
# TM outgoing edges plot
heatmap = np.empty((5, 2001))
heatmap = np.empty((5, len(UF_in_aggr_wdf['UF_*'])))
=======
######## UF out #########
heatmap = np.empty((5, len(UF_out_aggr_wdf['UF_*'])))
>>>>>>> dev-ab

for i, (column_name, column_data) in enumerate(UF_out_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

<<<<<<< HEAD
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(5))
ax.set_yticklabels(['UF_*', 'UF_UF', 'UF_UM', 'UF_TM', 'UF_TF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{dataset} UF source actor outgoing activity')
=======
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['UF_(UF,UM,TM,TF)', 'UF_UF', 'UF_UM', 'UF_TM', 'UF_TF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} UF source actor outgoing activity')
>>>>>>> dev-ab

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_UF_out_node_activity.png')

<<<<<<< HEAD
=======
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

>>>>>>> dev-ab




######## TF #########
TF_in_aggr_wdf = in_infl_weights_df[['TF_*', 'TF_TF', 'TF_TM', 'TF_UM', 'TF_UF']]
TF_out_aggr_wdf = out_infl_weights_df[['TF_*', 'TF_TF', 'TF_TM', 'TF_UM', 'TF_UF']]
<<<<<<< HEAD
TF_in_aggr_wdf = TF_out_aggr_wdf.loc[TF_out_aggr_wdf['TF_*'] !=0]
=======
TF_in_aggr_wdf = TF_in_aggr_wdf.loc[TF_in_aggr_wdf['TF_*'] !=0]
>>>>>>> dev-ab
TF_out_aggr_wdf = TF_out_aggr_wdf.loc[TF_out_aggr_wdf['TF_*'] !=0]

TF_out_aggr_wdf = TF_out_aggr_wdf.sort_values(by=['TF_*'], ascending=False)
TF_in_aggr_wdf = TF_in_aggr_wdf.sort_values(by=['TF_*'], ascending=False)

<<<<<<< HEAD
# TM outgoing edges plot
#heatmap = np.empty((5, 2001))
heatmap = np.empty((5, len(TF_in_aggr_wdf['TF_*'])))
=======
######## TF out #########
heatmap = np.empty((5, len(TF_out_aggr_wdf['TF_*'])))
>>>>>>> dev-ab

for i, (column_name, column_data) in enumerate(TF_out_aggr_wdf.items()):
    if(column_name!='actors'):
        heatmap[i] = column_data

<<<<<<< HEAD
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23)
ax.set_yticks(range(5))
ax.set_yticklabels(['TF_*', 'TF_TF', 'TF_TM', 'TF_UM', 'TF_UF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{dataset} TF source actor outgoing activity')
=======
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
im = ax.imshow(heatmap, interpolation='nearest', vmax=23, cmap='RdBu')
ax.set_yticks(range(5))
ax.set_yticklabels(['TF_(TF,TM,UM,UF)', 'TF_TF', 'TF_TM', 'TF_UM', 'TF_UF'])
ax.set_xlabel('rank of actors')
ax.set_title(f'{name} TF source actor outgoing activity')
>>>>>>> dev-ab

# colorbar
cbar = fig.colorbar(ax=ax, mappable=im, orientation = 'horizontal')
cbar.set_label('Transfer Entropy')

ax.set_aspect('auto')
plt.savefig(f'{dataset}_TF_out_node_activity.png')

<<<<<<< HEAD
=======
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
>>>>>>> dev-ab


