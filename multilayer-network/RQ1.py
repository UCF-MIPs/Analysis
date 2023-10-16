#TODO
# move generated dfs from node_correlation to results folder
# Get participation coeff working
# get ukr/skrip working simultaneously in node_correlation
# put plotting functions into loops
# put participation coeff calculations in loops
# move part coeff function to src
# isolate code for paper from the influence trees stuff

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from src import plot_heatmap
from src import plot_participant_coef

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

edge_types = ['TM_*', 'TF_*', 'UM_*', 'UF_*', '*_TM', '*_TF', '*_UM', '*_UF']
dataset = 'skrip_v7' # options: skrip_v7, ukr_v3

if(dataset=='ukr_v3'):
    name = 'Ukraine'
elif(dataset=='skrip_v7'):
    name = 'Skripal'

results_dir = 'multilayer-network/plots'

in_infl_wdf = pd.read_csv(f'multilayer-network/Data/preprocessed/{dataset}_in_infl_weights_df.csv')
out_infl_wdf = pd.read_csv(f'multilayer-network/Data/preprocessed/{dataset}_out_infl_weights_df.csv')

######## Heatmaps for each influence type #########

######## UM #########
## Sources/ Outgoing ##
UM_out_aggr_wdf = out_infl_wdf[['actors', 'UM_*', 'UM_UM', 'UM_UF', 'UM_TM', 'UM_TF']]
UM_out_aggr_wdf = UM_out_aggr_wdf.loc[UM_out_aggr_wdf['UM_*'] !=0]
UM_out_aggr_wdf = UM_out_aggr_wdf.sort_values(by=['UM_*'], ascending=False).dropna()
plot_heatmap.plot_heatmap_outdegree('UM', UM_out_aggr_wdf, name, results_dir)

## Targets/ Incoming ##
UM_in_aggr_wdf = in_infl_wdf[['actors', '*_UM', 'UM_UM', 'TM_UM', 'UF_UM', 'TF_UM']]
UM_in_aggr_wdf = UM_in_aggr_wdf.loc[UM_in_aggr_wdf['*_UM'] !=0]
UM_in_aggr_wdf = UM_in_aggr_wdf.sort_values(by=['*_UM'], ascending=False).dropna()
plot_heatmap.plot_heatmap_indegree('UM', UM_in_aggr_wdf, name, results_dir)



######## TM #########
## Sources/ Outgoing ##
TM_out_aggr_wdf = out_infl_wdf[['actors', 'TM_*', 'TM_TM', 'TM_UM', 'TM_TF', 'TM_UF']]
TM_out_aggr_wdf = TM_out_aggr_wdf.loc[TM_out_aggr_wdf['TM_*'] !=0]
TM_out_aggr_wdf = TM_out_aggr_wdf.sort_values(by=['TM_*'], ascending=False).dropna()
plot_heatmap.plot_heatmap_outdegree('TM', TM_out_aggr_wdf, name, results_dir)

## Targets/ Incoming ##
TM_in_aggr_wdf = in_infl_wdf[['actors', '*_TM', 'TM_TM', 'UM_TM', 'TF_TM', 'UF_TM']]
TM_in_aggr_wdf = TM_in_aggr_wdf.loc[TM_in_aggr_wdf['*_TM'] !=0]
TM_in_aggr_wdf = TM_in_aggr_wdf.sort_values(by=['*_TM'], ascending=False).dropna()
plot_heatmap.plot_heatmap_indegree('TM', TM_in_aggr_wdf, name, results_dir)



######## UF #########
## Sources/ Outgoing ##
UF_out_aggr_wdf = out_infl_wdf[['actors', 'UF_*', 'UF_UF', 'UF_TF', 'UF_UM', 'UF_TM']]
UF_out_aggr_wdf = UF_out_aggr_wdf.loc[UF_out_aggr_wdf['UF_*'] !=0]
UF_out_aggr_wdf = UF_out_aggr_wdf.sort_values(by=['UF_*'], ascending=False).dropna()
plot_heatmap.plot_heatmap_outdegree('UF', UF_out_aggr_wdf, name, results_dir)

## Targets/ Incoming ##
UF_in_aggr_wdf = in_infl_wdf[['actors', '*_UF', 'UF_UF', 'TF_UF', 'UM_UF', 'TM_UF']]
UF_in_aggr_wdf = UF_in_aggr_wdf.loc[UF_in_aggr_wdf['*_UF'] !=0]
UF_in_aggr_wdf = UF_in_aggr_wdf.sort_values(by=['*_UF'], ascending=False).dropna()
plot_heatmap.plot_heatmap_indegree('UF', UF_in_aggr_wdf, name, results_dir)



######## TF #########
## Sources/ Outgoing ##
TF_out_aggr_wdf = out_infl_wdf[['actors', 'TF_*', 'TF_TF', 'TF_UF', 'TF_TM', 'TF_UM']]
TF_out_aggr_wdf = TF_out_aggr_wdf.loc[TF_out_aggr_wdf['TF_*'] !=0]
TF_out_aggr_wdf = TF_out_aggr_wdf.sort_values(by=['TF_*'], ascending=False).dropna()
plot_heatmap.plot_heatmap_outdegree('TF', TF_out_aggr_wdf, name, results_dir)

## Targets/ Incoming ##
TF_in_aggr_wdf = in_infl_wdf[['actors', '*_TF', 'TF_TF', 'UF_TF', 'TM_TF', 'UM_TF']]
TF_in_aggr_wdf = TF_in_aggr_wdf.loc[TF_in_aggr_wdf['*_TF'] !=0]
TF_in_aggr_wdf = TF_in_aggr_wdf.sort_values(by=['*_TF'], ascending=False).dropna()
plot_heatmap.plot_heatmap_indegree('TF', TF_in_aggr_wdf, name, results_dir)

#################################
### participation coefficient ###
#################################

def part_coeff(df, sour_infl):
    # probably should include actors and go row by row
    m=4
    types = ['UF', 'UM', 'TF', 'TM']
    aggr_type = str(sour_infl + '_*')
    print(aggr_type)
    o = df[aggr_type].to_numpy()
    ptemp = np.zeros_like(o)
    for i in types:
        infl_type = str(sour_infl + '_' + i)
        print(infl_type)
        k = df[infl_type].to_numpy()
        #print(np.count_nonzero(o))
        #ptemp += np.divide(k,o,out=np.zeros_like(k), where=o!=0, dtype=float)**2
        ptemp += np.divide(k,o)**2
        #print(ptemp)
    ptemp[np.isinf(ptemp)] = np.nan
    p = (4./3.)*(1-ptemp)
    return p # array of part. coeffs

part_cof_UM = part_coeff(out_infl_wdf, 'UM')
part_cof_TM = part_coeff(out_infl_wdf, 'TM')
part_cof_UF = part_coeff(out_infl_wdf, 'UF')
part_cof_TF = part_coeff(out_infl_wdf, 'TF')

part_df = pd.DataFrame({'actors': out_infl_wdf['actors'],\
                        'p_UM': part_cof_UM,\
                        'p_TM': part_cof_TM,\
                        'p_UF': part_cof_UF,\
                        'p_TF': part_cof_TF})

part_df = part_df.dropna(how='all')
part_df = part_df[ \
            part_df['p_UM'].notna() | \
            part_df['p_TM'].notna() | \
            part_df['p_UF'].notna() | \
            part_df['p_TF'].notna() \
            ]


part_df.to_csv(f'multilayer-network/results/{dataset}_out_part_coef.csv')

### new participant coefficient functions ###
## Sources/outdegree ##
def part_coef_out(df, inf, layers):
    types = ['UM','TM','UF','TF']
    aggr_type = str(inf + '_*')
    df['ptemp'] = 0
    for i in types:
        infl_type = str(inf + '_' + i)
        df['ptemp'] += np.power(np.divide(df[infl_type],df[aggr_type]),2)
    df['pc_out'] = (layers/(layers-1))*(1-df['ptemp'])
    return df 

## Targets/Incoming ##
def part_coef_in(df, inf, layers):
    types = ['UM','TM','UF','TF']
    aggr_type = str('*_' + inf)
    df['ptemp'] = 0
    for i in types:
        infl_type = str(i + '_' + inf)
        df['ptemp'] += np.power(np.divide(df[infl_type],df[aggr_type]),2)
    df['pc_in'] = (layers/(layers-1))*(1-df['ptemp'])
    return df 

## Creating Plots for Participant Coefficient of Sources ##
part_coef_out(UM_out_aggr_wdf,'UM', 4)
part_coef_out(TM_out_aggr_wdf,'TM', 4)
part_coef_out(UF_out_aggr_wdf,'UF', 4)
part_coef_out(TF_out_aggr_wdf,'TF', 4)
plt.clf()
plot_participant_coef.part_coef_plot_out('UM', UM_out_aggr_wdf, name, results_dir)
plt.clf()
plot_participant_coef.part_coef_plot_out('TM', TM_out_aggr_wdf, name, results_dir)
plt.clf()
plot_participant_coef.part_coef_plot_out('UF', UF_out_aggr_wdf, name, results_dir)
plt.clf()
plot_participant_coef.part_coef_plot_out('TF', TF_out_aggr_wdf, name, results_dir)

## Creating Plots for Participant Coefficient of Targets ##
part_coef_in(UM_in_aggr_wdf,'UM', 4)
part_coef_in(TM_in_aggr_wdf,'TM', 4)
part_coef_in(UF_in_aggr_wdf,'UF', 4)
part_coef_in(TF_in_aggr_wdf,'TF', 4)
plt.clf()
plot_participant_coef.part_coef_plot_in('UM', UM_in_aggr_wdf, name, results_dir)
plt.clf()
plot_participant_coef.part_coef_plot_in('TM', TM_in_aggr_wdf, name, results_dir)
plt.clf()
plot_participant_coef.part_coef_plot_in('UF', UF_in_aggr_wdf, name, results_dir)
plt.clf()
plot_participant_coef.part_coef_plot_in('TF', TF_in_aggr_wdf, name, results_dir)


### COMPARING ALL AGGREGATES
#TODO finish this
'''
# Shows comparitive heatmap of activity of influence sources vs. audience

#in_aggr_infl_wdf = in_infl_wdf[['UM_*', 'UF_*', 'TM_*', 'TF_*']]
#out_aggr_infl_wdf = out_infl_wdf[['UM_*', 'UF_*', 'TM_*', 'TF_*']]

# To include plot of corresponding index of in to out plot and out to in plot:
#https://stackoverflow.com/questions/58423707/pandas-matching-index-of-one-dataframe-to-the-column-of-other-dataframe
# Would show that outgoing actors are/are not the same as incoming

# sort by value of first column

out_aggr_infl_weights_df = out_aggr_infl_wdf.sort_values(by=['UM_*'], ascending=False)
in_aggr_infl_weights_df = in_aggr_infl_wdf.sort_values(by=['UM_*'], ascending=False)

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

