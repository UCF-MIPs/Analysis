from bluebox import *
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns


#print("Pre-Processing - merging URL list and Majestic Million")
news_df.rename(columns = {'news outlets':'Domain'}, inplace=True)
pop_df.rename(columns = {'RefSubNets':'pop_score'}, inplace=True)
final_df = pd.merge(news_df, pop_df, on='Domain', how='left')
final_df = final_df[['ID','Domain','pop_score']]

#print("Pre-Processing - merging NewsGuard to table")
trust_df = trust_df[['Domain','Score']]
trust_df.rename(columns = {'Score':'trust_score'}, inplace=True)
# left join final_df and trust scores
final_df = pd.merge(final_df, trust_df, on='Domain', how='left')
final_df = cutoff(final_df, 'pop_score')
final_df = reg_normalize(final_df, 'pop_score')
final_df['pop_score'] = final_df['pop_score'].round(decimals=1)
#print("Classifying TUFM")
final_df = select_TUFM(final_df, trust_cutoff, pop_cutoff)




TM = final_df[final_df['tufm_class'] == 'TM']
TF = final_df[final_df['tufm_class'] == 'TF']
UM = final_df[final_df['tufm_class'] == 'UM']
UF = final_df[final_df['tufm_class'] == 'UF']
'''
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

'''

df = final_df.drop(['tufm_class','ID','Domain'],axis=1)
print(df)

sns.scatterplot(x="trust_score", y="pop_score", data=df)
plt.xlabel("X", size=16)
plt.ylabel("y", size=16)
plt.title("Scatter Plot with Seaborn", size=18)
plt.savefig("simple_scatter_plot_Seanborn.png")



sns.jointplot(x="trust_score",
              y="pop_score",
             edgecolor="white",
             data=df);
#plt.title("Scatter Plot with Marginal Histograms: Seaborn", size=18, pad=80)
plt.savefig("marginal_plot_Seaborn.png")
