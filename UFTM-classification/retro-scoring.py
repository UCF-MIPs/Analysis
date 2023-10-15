import pandas as pd
import boto3
import io
import os

def load_df(year = '2020', month = '01', day = '01', hour = '00'):
    """
    Read all time separated files from NewsGuard s3 bucket
    """
    filename = str('full-metadata/' + year + '/' + month + '/' + 'metadata-' + year+month+day+hour + '.csv') 
    bucket, filename = "newsguard-feeds", filename
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, filename)
    with io.BytesIO(obj.get()['Body'].read()) as bio:
        df = pd.read_csv(bio)
        df = df.drop(df[df['Country']=='ALL'].index)
    return df

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
years  = ['2019', '2020', '2021', '2022']
days   = ['01', '15']

data_dir        = "data"
results_dir     = "results"
results_fname   = os.path.join(results_dir + "history.csv")
news_filename   = "news_outlets.xlsx"
news_data_path  = os.path.join(data_dir + "/" + news_filename)
news_df         = pd.read_excel(news_data_path)
column_names    = news_df['news outlets']

datelist=[]
history_df = pd.DataFrame(columns = column_names)
for year in years:
    for month in months:
        datelist.append(str(year+month))

history_df.insert(loc=0, column='date', value = datelist)

for year in years:
    print("year: " + year)
    for month in months:
        print("month: " + month)
        if (year == '2019' and (month == '01' or month =='02')) or (year == '2022' \
                and (month == '09' or month =='10' or month=='11' or month=='12')) or \
                (year == '2020' and month =='08'):
            pass
        else:
            df = load_df(year=year, month=month)
            for url in history_df.columns[1:]:
                if df['Domain'].str.contains(url).any():
                    datetime=(year+month)
                    print(datetime)
                    print(url)
                    mask = history_df['date'] == str(year+month)
                    new_score = df.loc[df['Domain'] == url, 'Score'].to_string(index=False)
                    print(new_score)
                    history_df.loc[mask, url] = new_score
                    print(history_df)
                else:
                    print("no URL match")
                
history_df.to_csv(results_fname, index=False)

