import os
import tweepy
import pandas as pd
import numpy as np
import time

from termcolor import colored


class tweepyscraping:

    def __init__(self):
        self.twitter_api_initialization()
    
    # Twitter API
    def twitter_api_initialization(self):
        # Twitter API initialization
        consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
        consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
        access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        self.api = api

    # get tweet using tweepy
    def __gettweet(self, query, count):
        api = self.api
        text_query = query+" -filter:retweets"
        tweets = []
        try:
            # pulling individual tweets from query
            for tweet in api.search(q=text_query, count=count, tweet_mode="extended"):
                # adding to list that contains "tweet.user.screen_name, tweet.id, tweet.text"
                tweets.append((tweet.user.screen_name, tweet.id, tweet.full_text))
                # creation of dataframe from tweets list
                tweetsdf = pd.DataFrame(tweets,columns=['screen_name', 'tweet_id', 'text'])
                # dataframe index start from 1
                tweetsdf.index = np.arange(1,len(tweetsdf)+1)
            print(colored("Tweets collected, query='{0}', count='{1}'".format(query, count), "green"))
            return tweetsdf
        except BaseException as e:
            print('failed on_status,', str(e))
            time.sleep(3)
            return None
    
    # public, get tweet
    def get_tweet(self, query, count):
        return self.__gettweet(query, count)