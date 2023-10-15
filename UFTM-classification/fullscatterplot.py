import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
from bluebox import select_TUFM, log_normalize, reg_normalize
import os
import pandas as pd
import numpy as np

# dir for results output, an empty dir needs to already exist. If not empty, will overwrite.
results_dir      = "results"
data_dir         = "data"
new_csv_filename = "news_table-v3-UT60-FM5.csv"

news_filename    = "news_outlets.xlsx"
pop_filename     = "majestic_million.csv"
trust_filename   = "metadata-2023041709.csv"
trust_cutoff     = 60
pop_cutoff       = 5

news_data_path  = os.path.join(data_dir + "/" + news_filename)
news_df         = pd.read_excel(news_data_path)
pop_data_path   = os.path.join(data_dir + "/" + pop_filename)
pop_df          = pd.read_csv(pop_data_path)
trust_data_path = os.path.join(data_dir + "/" + trust_filename)
trust_df        = pd.read_csv(trust_data_path)
results_path    = os.path.join(results_dir + "/" + new_csv_filename)

pop_df.rename(columns = {'RefSubNets':'pop_score'}, inplace=True)
pop_df = pop_df[['Domain','pop_score']]
final_df = pop_df
final_df = final_df[['Domain','pop_score']]
trust_df = trust_df[['Domain','Score','Country','Language']]
trust_df.rename(columns = {'Score':'trust_score'}, inplace=True)
final_df = pd.merge(final_df, trust_df, on='Domain', how='left')
final_df['pop_score'].replace('', np.nan, inplace=True)
final_df['trust_score'].replace('', np.nan, inplace=True)
final_df.dropna(inplace=True)
#final_df = log_normalize(final_df, 'pop_score')
final_df = reg_normalize(final_df, 'pop_score')
final_df['pop_score'] = final_df['pop_score'].round(decimals=1)
#final_df = select_TUFM(final_df, trust_cutoff, pop_cutoff_en, pop_cutoff_glob)
final_df = select_TUFM(final_df, trust_cutoff, pop_cutoff)



TM = final_df[final_df['tufm_class'] == 'TM']
TF = final_df[final_df['tufm_class'] == 'TF']
UM = final_df[final_df['tufm_class'] == 'UM']
UF = final_df[final_df['tufm_class'] == 'UF']

# MAIN PLOT
plt.scatter(TM.trust_score, TM.pop_score, c='b', label = 'Trustworthy-Mainstream')
plt.scatter(TF.trust_score, TF.pop_score, c='r', label = 'Trustworthy-Fringe')
plt.scatter(UM.trust_score, UM.pop_score, c='g', label = 'Untrustworthy-Mainstream')
plt.scatter(UF.trust_score, UF.pop_score, c='y', label = 'Untrustworthy-Fringe')

legend1 = plt.legend(loc='upper left', bbox_to_anchor=(1, 0.9))
plt.gca().add_artist(legend1)

# SINGLE POINTS - trust_score, pop_score, classflag
# time.com - TM
a = (100, 47.6)
# weather.com - TM
a1 = (95, 21.3)
# gainesvilletimes.com - TF
b = (100, 1.1)
# telemundo.com - TF
b1 = (95, 2.3)
# rt.com - UM
c = (24.2, 12.5)
# breitbart.com - UM
c1 = (49.5, 12.4)
# lifenews.com - UF
d = (30, 2.6)
# drudgereport.com - UF AT 60, THIS IS T, not U
#d1 = (62.5, 3.7)
# judicialwatch.org
d1 = (57,2.7)

points = [a,a1,b,b1,c,c1,d,d1]
color = ['b', 'b', 'r', 'r', 'g', 'g', 'y', 'y']
marks = ['P', 'X', 'P', 'X', 'P', 'X', 'P', 'X']
points_arr = np.array(points)
legend2_names = ["time.com", "weather.com", "gainesvilletimes.com", "telemundo.com", "rt.com", "breitbart.com", "lifenews.com", "judicialwatch.org"]
examples = []
for i in range(len(points_arr[:,0])):
    examples.append(plt.scatter(points_arr[i,0], points_arr[i,1], c=color[i], edgecolors='black', marker = marks[i], s=150, label = legend2_names[i]))

plt.legend(handles = examples,loc='lower left', bbox_to_anchor=(1, 0.05))
plt.xlabel('Trustworthiness Score')
plt.ylabel('Popularity Score')
plt.title('News Sources')
plt.vlines(x=60, ymin=0, ymax=100, linestyles='dashed', colors='black',linewidth=3.0)
plt.hlines(y=5, xmin=0, xmax=100, linestyles='dashed', colors='black',linewidth=3.0)
results_path = os.path.join(results_dir + "/" + "scatter.png")
plt.savefig(results_path, bbox_inches="tight")



