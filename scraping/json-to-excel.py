import pandas as pd

# columns = ['screen_name','username','user_id','tweet_id','tweet_url','timestamp','timestamp_epochs','text','text_html','links','hashtags','has_media','img_urls','video_url','likes','retweets','replies','is_replied','is_reply_to','parent_tweet_id','reply_to_users']
columns = ['username','user_id','tweet_url','timestamp','timestamp_epochs','text_html','links','hashtags','has_media','img_urls','video_url','likes','retweets','replies','is_replied','is_reply_to','parent_tweet_id','reply_to_users']

dataset = pd.read_json('radikal-negative.json', encoding="utf8")
newdata = dataset.drop(columns, axis='columns')
newdata.to_excel('radikal-negative.xlsx', index=False)

