from flask import render_template, url_for, request

import pandas as pd
import numpy as np

# local package (package gawean dewe)
from scraping.tweepyscraping import tweepyscraping
from model.preprocessing import preprocessing
from model.lstm import lstm_pretrained

from termcolor import colored


# public app class
class app_public:

    def __init__(self):
        self.tweetscrap = tweepyscraping()
        self.preprocess = preprocessing()
        self.predict = lstm_pretrained()

    # 404 not found
    def page_not_found(self, e):
        return render_template('public/404.html'), 404

    # index page
    def index(self):
        return render_template('public/index.html')

    # about page
    def about(self):
        return render_template('public/about.html')

    # klasifikasi teks page
    def klasifikasi_text(self):

        if request.method  == 'POST':
            try:
                text = request.form['text']
                if not text:
                    raise ValueError('Empty Text')
                clean_text = self.preprocess.clean_text(text)
                preprocessed_text = clean_text['preprocessed_text'].tolist()
                prediction, probability = self.predict.predict_text(preprocessed_text)

                print(colored("Summary : '{0}', '{1}', '{2}'".format(prediction, probability, preprocessed_text), "green"))
                return render_template('public/klasifikasi_text.html', status=True,
                                                                    text=text,
                                                                    prediction=prediction, 
                                                                    probability=probability
                                                                    )
            except:
                return render_template('public/klasifikasi_text.html', status=False)
        else:
            return render_template('public/klasifikasi_text.html', status=False)

    # klasifikasi tweet page
    def klasifikasi_tweet(self):

        if request.method  == 'POST':
            try:
                text = request.form['twitter-search']
                tweet_count = int(request.form['tweet-count'])
                if not text:
                    raise ValueError('Empty Text')

                tweets = self.tweetscrap.get_tweet(text, tweet_count)
                tweets = self.preprocess.clean_tweet(tweets)
                preprocessed_tweets = tweets['preprocessed_text'].tolist()
                prediction, prediction_count = self.predict.predict_tweets(preprocessed_tweets)

                pd.set_option('display.max_colwidth', None)
                tweets['prediction'] = prediction
                total_tweet = len(tweets.index)
                tweets = tweets.drop(['tweet_id','preprocessed_text'], axis=1)
                tweets.columns = ['Username','Tweet','Prediksi']
                table = tweets.to_html().replace('<table','<table class="table dataframe table-bordered" id="dataTable" width="100%" cellspacing="0"')

                print(colored("Summary : '{0}', '{1}', '{2}'".format(text, total_tweet, prediction_count), "green"))
                return render_template('public/klasifikasi_tweet.html', status = True,
                                                                    keyword=text,
                                                                    table1=table,
                                                                    total_tweet=total_tweet,
                                                                    prediction_count=prediction_count
                                                                    )
            except:
                return render_template('public/klasifikasi_tweet.html', status=False)
        else:
            return render_template('public/klasifikasi_tweet.html', status=False)

    # pengujian page
    def tentang_pengujian(self):
        try:
            report = pd.read_excel('data/result_history/_validation_report.xlsx')
            hasil = report[["accuracy", "precision", "recall", "f1"]].mean(axis=0)
            hasil = round(hasil,2)
            return render_template('public/tentang_pengujian.html', status = True, hasil1 = hasil)
        except:
            return render_template('public/tentang_pengujian.html', status=False)

    # model page
    def tentang_model(self):
        return render_template('public/tentang_model.html')

#test section
    # test page, tesing purpose only
    def test(self):
        if request.method  == 'POST':
            return render_template('public/test.html')
        else:
            return render_template('public/test.html')

    # bunch scraping, tesing purpose only
    def dataset_scraping(self):
        try:
            query = pd.read_excel ('raw-data/dataset_query.xlsx')
            print(query)

            data_tweet = []
            for index, row in query.iterrows():
                print(index)
                temp = self.tweetscrap.get_tweet(row['query'], 20)
                data_tweet.append(temp)
            
            dataset = pd.concat(data_tweet)

            print(dataset)

            dataset.to_excel('raw-data/additional_dataset.xlsx')


            return render_template('public/index.html')
        except:
            return render_template('public/index.html')

    # klasifikasi data page
    def klasifikasi_data(self):

        try:
            data = pd.read_excel('raw-data/extra_dataset.xlsx')

            data = self.preprocess.clean_tweet(data)
            preprocessed_data = data['preprocessed_text'].tolist()
            prediction, _ = self.predict.predict_tweets(preprocessed_data)

            data['prediction'] = prediction
            print(data)

            data.to_excel('raw-data/plus_dataset.xlsx')

            return render_template('public/index.html')
        except:
            return render_template('public/index.html')