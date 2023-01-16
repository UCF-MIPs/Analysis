import pandas as pd
#pd.set_option('display.max_colwidth', None)
import glob
import os

path = 'data' # All data that generated TE network as input
all_files = glob.glob(os.path.join(path , "*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=5)
    li.append(df)

x = pd.concat(li, axis=0, ignore_index=True)
x.drop_duplicates()

col_to_keep = ['Twitter Retweet of', 'Thread Author', 'Original Url', 'Full Name', 'Author', 'Domain', \
        'Url', 'Title', 'Display URLs', 'Expanded URLs', 'Thread Author', 'Full Text']

"""
list of all fields

['Query Id', 'Query Name', 'Date', 'Title', 'Url', 'Domain', 'Sentiment', 'Page Type', 'Language', 'Country Code', 'Continent Code', 'Continent', 'Country', 'City Code', 'Account Type', 'Added', 'Assignment', 'Author', 'Avatar', 'Category Details', 'Checked', 'City', 'Display URLs', 'Expanded URLs', 'Facebook Author ID', 'Facebook Comments', 'Facebook Likes', 'Facebook Role', 'Facebook Shares', 'Facebook Subtype', 'Full Name', 'Full Text', 'Gender', 'Hashtags', 'Impact', 'Impressions', 'Instagram Comments', 'Instagram Followers', 'Instagram Following', 'Instagram Interactions Count', 'Instagram Likes', 'Instagram Posts', 'Interest', 'Last Assignment Date', 'Latitude', 'Location Name', 'Longitude', 'Media Filter', 'Media URLs', 'Mentioned Authors', 'Original Url', 'Priority', 'Professions', 'Resource Id', 'Short URLs', 'Starred', 'Status', 'Subtype', 'Thread Author', 'Thread Created Date', 'Thread Entry Type', 'Thread Id', 'Thread URL', 'Total Monthly Visitors', 'Twitter Author ID', 'Twitter Channel Role', 'Twitter Followers', 'Twitter Following', 'Twitter Reply Count', 'Twitter Reply to', 'Twitter Retweet of', 'Twitter Retweets', 'Twitter Tweets', 'Twitter Verified', 'Updated', 'Reach (new)', 'Air Type', 'Blog Name', 'Broadcast Media Url', 'Broadcast Type', 'Content Source', 'Content Source Name', 'Copyright', 'Engagement Type', 'Is Syndicated', 'Item Review', 'Linkedin Comments', 'Linkedin Engagement', 'Linkedin Impressions', 'Linkedin Likes', 'Linkedin Shares', 'Linkedin Sponsored', 'Linkedin Video Views', 'Media Type', 'Page Type Name', 'Parent Blog Name', 'Parent Post Id', 'Pub Type', 'Publisher Sub Type', 'Rating', 'Reddit Author Awardee Karma', 'Reddit Author Awarder Karma', 'Reddit Author Karma', 'Reddit Comments', 'Reddit Score', 'Reddit Score Upvote Ratio', 'Region', 'Region Code', 'Root Blog Name', 'Root Post Id', 'Subreddit', 'Subreddit Subscribers', 'Weblog Title']
"""

x = x[col_to_keep]

substring = 'washingtonpost'
x = x[x.apply(lambda row: row.astype(str).str.contains(substring, case=False).any(), axis=1)]

#x = x.loc[x['Author']==substring]
y = x['Title'].values.tolist()
#print(y[0])

x.to_csv('TE_washpost_results.csv')
