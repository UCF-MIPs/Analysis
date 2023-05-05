import numpy as np
import pandas as pd
import os

#   Script to generate trustworthiness/popularity table for news sources

#   ### Functions ###
#TODO move to separate files

def mean_normalize(df, column_name):
    existing_range = df[column_name].max()-df[column_name].min()
    data = df[column_name]
    x    = (data - data.std())/existing_range
    new_column_name = column_name + '_normalized'
    df[new_column_name] = x
    return df

def rank2score(df, column_name):
    """
    Inverts value of rankings to represent a magnitude score
    """ 
    df[column_name] = df[column_name].max()-df[column_name]
    return df

def cutoff(df, column_name):
    # exclude outliers, range is roughly 1-80,000 setting 80,000 as max
    df.loc[df[column_name] > 160000, column_name] = 160000
    return df

def reg_normalize(df, column_name):
    """
    Normalize from 0 to 100
    """
    #TODO handle bad/empty values, either here or check input file
    #normalize
    data = df[column_name]
    data = df[column_name] - data.min()
    existing_range = data.max()-data.min()
    x  = data/existing_range
    #new_column_name = column_name + '_normalized'
    df[column_name] = x*100
    return df

def reg_normalize2(df, column_name):
    """
    Normalize from 0 to 100
    """
    #TODO handle bad/empty values, either here or check input file
    #normalize
    data = df[column_name]
    existing_range = data.max()-data.min()
    x  = data/existing_range
    #new_column_name = column_name + '_normalized'
    df[column_name] = x*100
    return df


def select_TUFM(df, trust_cutoff=50, pop_cutoff_en=50, pop_cutoff_glob=50):
    """
    A function to classify a news source based on reputation and popularity score.

    Trustworthy (T) or Untrustworthy (U) 
    and
    Fringe (F) or Mainstream (M)
    """

    df_en = df[df['Language'] == 'en']
    df_glob = df[df['Language'] != 'en']

    conditions_en = [ 
            (df_en['trust_score'] >= trust_cutoff) & (df_en['pop_score'] >= pop_cutoff_en), #TM 
            (df_en['trust_score'] >= trust_cutoff) & (df_en['pop_score'] < pop_cutoff_en), #TF 
            (df_en['trust_score'] < trust_cutoff)  & (df_en['pop_score'] >= pop_cutoff_en), #UM 
            (df_en['trust_score'] < trust_cutoff)  & (df_en['pop_score'] < pop_cutoff_en), #UF 
    ]
    conditions_glob = [ 
            (df_glob['trust_score'] >= trust_cutoff) & (df_glob['pop_score'] >= pop_cutoff_glob), #TM 
            (df_glob['trust_score'] >= trust_cutoff) & (df_glob['pop_score'] < pop_cutoff_glob), #TF 
            (df_glob['trust_score'] < trust_cutoff)  & (df_glob['pop_score'] >= pop_cutoff_glob), #UM 
            (df_glob['trust_score'] < trust_cutoff)  & (df_glob['pop_score'] < pop_cutoff_glob), #UF 
    ]
    categories = ['TM','TF', 'UM', 'UF']
    df_en['tufm_class'] = np.select(conditions_en, categories)
    df_glob['tufm_class'] = np.select(conditions_glob, categories)
    df = pd.concat([df_en, df_glob])
    return df

#   ### Default directory structure ###

# dir for results output, an empty dir needs to already exist. If not empty, will overwrite.
results_dir      = "results"
data_dir         = "data"
new_csv_filename = "news_table-v3-UT60-FM5.csv"


#   ### User input ###

# news data
news_filename    = "news_outlets.xlsx"
pop_filename     = "majestic_million.csv"
trust_filename   = "metadata-2023041709.csv"
trust_cutoff     = 60
pop_cutoff_en    = 5
pop_cutoff_glob  = 0.75

#   ### Create dataframes ###

news_data_path  = os.path.join(data_dir + "/" + news_filename)
news_df         = pd.read_excel(news_data_path)
pop_data_path   = os.path.join(data_dir + "/" + pop_filename)
pop_df          = pd.read_csv(pop_data_path)
trust_data_path = os.path.join(data_dir + "/" + trust_filename)
trust_df        = pd.read_csv(trust_data_path)

#   ### Execution ###
if __name__ == '__main__':
    print("Checking values")
    #TODO check column values for empties, invalid values

    print("Pre-Processing - merging URL list and Majestic Million")
    # Join Majestic Million data to URL list
    # rename URL list column name to match majestic millions column name
    news_df.rename(columns = {'news outlets':'Domain'}, inplace=True)
    news_df = news_df[['Domain']] 
    #rename column in pop_df
    pop_df.rename(columns = {'RefSubNets':'pop_score'}, inplace=True)
    # Keep only needed columns in pop_df
    pop_df = pop_df[['Domain','pop_score']]
    # left join URL list and majestic million (popularity scores)
    final_df = pop_df
    #remove unnecessary columns from majestic million, just keep global rankings
    final_df = final_df[['Domain','pop_score']]

    print("Pre-Processing - merging NewsGuard to table")
    # NewsGuard data
    trust_df = trust_df[['Domain','Score','Country','Language']]
    trust_df.rename(columns = {'Score':'trust_score'}, inplace=True)
    # left join final_df and trust scores
    final_df = pd.merge(final_df, trust_df, on='Domain', how='left')
    final_df = pd.merge(news_df, final_df, on='Domain', how='left')
    

    # Add other languages (outside of 1000 news sources provided)
    multilang_df = trust_df[trust_df['Language'] != ('en' or 'es')]

    multilang_df = pd.merge(multilang_df, pop_df, on='Domain', how='left')
    final_df = pd.concat([final_df,multilang_df]).drop_duplicates().reset_index(drop=True) 

    print(final_df)

    print("Normalizing columns")
    #final_df = cutoff(final_df, 'pop_score')
    #final_df = reg_normalize(final_df, 'trust_score')
    final_df = reg_normalize(final_df, 'pop_score') 
    final_df['pop_score'] = final_df['pop_score'].round(decimals=1)
    

    ### Plots ###
    
    '''
    print("global popularity distribution plot")
    import matplotlib.pyplot as plt
    #final_df['pop_score'].sort_values(ignore_index=True).plot()
    globpop = final_df[final_df['Language'] != 'en']
    globpop = globpop.dropna()
    print(globpop)
    y_1 = np.linspace(0,max(globpop.pop_score), len(globpop.pop_score))
    plt.plot(globpop['pop_score'].sort_values(ignore_index=True), y_1)
    plt.ylabel("News Sources")
    plt.xlabel("Popularity Score")
    plt.title("News Source Popularity")
    #plt.text(.5, .0001, "Source: Majestic Million", ha='center')
    #plt.text(.5, .800, "Source: Majestic Million", ha='center')
    plt.vlines(x=0.75, ymin=0, ymax=100, linestyles='dashed', label = "Popular-Fringe Threshold")
    plt.legend()
    plt.savefig('pop_score_dist.png')
    '''
    
    
    print("popularity distribution plot")
    import matplotlib.pyplot as plt
    plt.clf()
    #final_df['trust_score'].sort_values(ignore_index=True).plot(kind="bar")
    bins = np.arange(0,5,0.25)
    globpop = final_df[final_df['Language'] != 'en']
    globpop = globpop.dropna()
    #final_df['pop_score'].hist(grid=False, bins=bins)
    globpop['pop_score'].hist(grid=False, bins=bins)
    plt.xticks(bins[::4])
    plt.ylabel("Number of News Sources")
    plt.xlabel("Popularity Score - global")
    plt.title("News Source Popularity (Source: Majestic Million)")
    #txt = "Source: Majestic Million"
    #plt.figtext(0.5, 0.05, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.vlines(x=0.75, ymin=0, ymax=425, linestyles='dashed', colors='black', label = "Fringe-Popular Threshold")
    plt.legend()
    plt.savefig('pop_score_dist.png')
    


    '''
    print("trustworthiness distribution plot")
    plt.clf()
    #final_df['trust_score'].sort_values(ignore_index=True).plot(kind="bar")
    bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    final_df['trust_score'].hist(grid=False, bins=bins)
    plt.xticks(bins)
    plt.ylabel("Number of News Sources")
    plt.xlabel("Trustworthiness Score")
    plt.title("News Source Trustworthiness (Source: NewsGuard)")
    #txt = "Source: NewsGuard"
    #plt.figtext(0.5, 0.05, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.vlines(x=60, ymin=0, ymax=550, linestyles='dashed', colors='black', label = "Untrustworthy-Trustworthy Threshold")
    plt.legend()
    plt.savefig('trust_score_dist.png')
    '''

    print("Classifying TUFM")
    final_df = select_TUFM(final_df, trust_cutoff, pop_cutoff_en, pop_cutoff_glob)
   
    print("Drop rows with incomplete data")
    final_df = final_df[final_df.tufm_class != str(0)]

    #print("Drop duplicate rows")
    #final_df = final_df.drop_duplicates()
    print("Generating new file in /results")
    results_path = os.path.join(results_dir + "/" + new_csv_filename)
    final_df.to_csv(results_path, index=False)


