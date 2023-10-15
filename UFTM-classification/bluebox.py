import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from util import dist_bar_plot, dist_curve_plot, joint_plot 
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


def log_normalize(df, column_name):
    """
    Normalize from 0 to 100 based on the log of input value from column_name
    """
    #TODO handle bad/empty values, either here or check input file
    #normalize
    data = df[column_name]
    existing_range = data.max()-data.min()
    x = data/existing_range
    x = x*100
    x  = np.log(x + 1)
    #new_column_name = column_name + '_normalized'
    df[column_name] = x
    return df





def select_TUFM(df, trust_cutoff=50, pop_cutoff=50):
    """
    A function to classify a news source based on reputation and popularity score.

    Trustworthy (T) or Untrustworthy (U) 
    and
    Fringe (F) or Mainstream (M)
    """

    conditions = [ 
            (df['trust_score'] >= trust_cutoff) & (df['pop_score'] >= pop_cutoff), #TM 
            (df['trust_score'] >= trust_cutoff) & (df['pop_score'] < pop_cutoff), #TF 
            (df['trust_score'] < trust_cutoff)  & (df['pop_score'] >= pop_cutoff), #UM 
            (df['trust_score'] < trust_cutoff)  & (df['pop_score'] < pop_cutoff), #UF 
    ]
    categories = ['TM','TF', 'UM', 'UF']
    df['tufm_class'] = np.select(conditions, categories)
    return df



def select_TUFM_seperatate_english(df, trust_cutoff=50, pop_cutoff_en=50, pop_cutoff_glob=50):
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
#pop_cutoff_en    = 5
#pop_cutoff_glob  = 0.75
pop_cutoff       = 5

#   ### Create dataframes ###

# Not currently using provided news source list

#news_data_path  = os.path.join(data_dir + "/" + news_filename)
#news_df         = pd.read_excel(news_data_path)
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
    #news_df.rename(columns = {'news outlets':'Domain'}, inplace=True)
    #news_df = news_df[['Domain']] 

    #rename column in pop_df
    pop_df.rename(columns = {'RefSubNets':'pop_score'}, inplace=True)
    # Keep only needed columns in pop_df
    pop_df = pop_df[['Domain','pop_score']]
    # left join URL list and majestic million (popularity scores)
    # final_df = pd.merge(news_df, pop_df, on='Domain', how='left')
    final_df = pop_df
    #remove unnecessary columns from majestic million, just keep global rankings
    final_df = final_df[['Domain','pop_score']]

    print("Pre-Processing - merging NewsGuard to table")
    # NewsGuard data
    trust_df = trust_df[['Domain','Score','Country','Language']]
    trust_df.rename(columns = {'Score':'trust_score'}, inplace=True)
    # left join final_df and trust scores
    final_df = pd.merge(final_df, trust_df, on='Domain', how='left')
    #final_df = pd.merge(news_df, final_df, on='Domain', how='left')
    

    # Add other languages (outside of 1000 news sources provided)
    #multilang_df = trust_df[trust_df['Language'] != ('en' or 'es')]

    #multilang_df = pd.merge(multilang_df, pop_df, on='Domain', how='left')
    #final_df = pd.concat([final_df,multilang_df]).drop_duplicates().reset_index(drop=True) 

    print("Removing rows with empty values") 
    final_df['pop_score'].replace('', np.nan, inplace=True)
    final_df['trust_score'].replace('', np.nan, inplace=True)
    final_df.dropna(inplace=True)

    print("Normalizing columns")
    final_df = log_normalize(final_df, 'pop_score') 
    final_df['pop_score'] = final_df['pop_score'].round(decimals=1)
    
    print("Classifying TUFM")
    #final_df = select_TUFM(final_df, trust_cutoff, pop_cutoff_en, pop_cutoff_glob)
    final_df = select_TUFM(final_df, trust_cutoff, pop_cutoff)

    print("Drop rows with incomplete data")
    #final_df = final_df[final_df.tufm_class != str(0)]
    #print("Drop duplicate rows")
    #final_df = final_df.drop_duplicates()
    #final_df.dropna() 

    print("Seperate news sources into english and non-english (italian, german, french)")
    df_en    = final_df[final_df['Language'] == 'en']
    df_nonen = final_df[final_df['Language'] != ('en' or 'es')]
    # spanish excluded due to of lack of data from NewsGuard
    
    ### Plots ###    
    print("Distribution bar plots")
    dist_bar_plot(df=df_en, metric = 'pop_score', cutoff_line=None, name='pop_en_dist_bar')
    dist_bar_plot(df=df_en, metric = 'trust_score', cutoff_line=None, name='trust_en_dist_bar')
    dist_bar_plot(df=df_nonen, metric = 'pop_score', cutoff_line=None, name='pop_nonen_dist_bar')
    dist_bar_plot(df=df_nonen, metric = 'trust_score', cutoff_line=None, name='trust_nonen_dist_bar')


    print("Distribution curve plots")
    dist_curve_plot(df=df_en, metric = 'pop_score', cutoff_line=None, name='pop_en_dist_bar')
    dist_curve_plot(df=df_en, metric = 'trust_score', cutoff_line=None, name='trust_en_dist_bar')
    dist_curve_plot(df=df_nonen, metric = 'pop_score', cutoff_line=None, name='pop_nonen_dist_bar')
    dist_curve_plot(df=df_nonen, metric = 'trust_score', cutoff_line=None, name='trust_nonen_dist_bar')

    print('seaborn distribution plot, trust vs pop')
    joint_plot(df_en, 'joint_plot_en')
    joint_plot(df_nonen, 'joint_plot_nonen')


    ### Splits ###
    print("Getting non-US splits\n")
    total = len(df_nonen)
    print(f"total non-english sources: {total}")
    fringe_perc = len(df_nonen[df_nonen['pop_score'] < pop_cutoff]) / total 
    print(f"fringe non-english sources: {fringe_perc}%")
    mainstream_perc = len(df_nonen[df_nonen['pop_score'] >= pop_cutoff]) / total
    print(f"mainstream non-english sources: {mainstream_perc}%")

    print("Getting english splits\n")
    total = len(df_en)
    print(f"total english sources: {total}")
    fringe_perc = len(df_en[df_en['pop_score'] < pop_cutoff]) / total 
    print(f"fringe english sources: {fringe_perc}%")
    mainstream_perc = len(df_en[df_en['pop_score'] >= pop_cutoff]) / total
    print(f"mainstream english sources: {mainstream_perc}%")
    
    print("Generating new file in /results")
    results_path = os.path.join(results_dir + "/" + new_csv_filename)
    results_path2 = os.path.join(results_dir + "/EN_" + new_csv_filename)
    results_path3 = os.path.join(results_dir + "/NON_EN" + new_csv_filename)
    final_df.to_csv(results_path, index=False)
    df_en.to_csv(results_path2, index=False)
    df_nonen.to_csv(results_path3, index=False)

