import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle
import pandas as pd
import numpy as np

from termcolor import colored


class lstm_pretrained:

    def __init__(self):
        self.__myload_model()
        self.__load_tokenizer()
        print(colored("Model initialized", "green"))

    # load tokenizer
    def __load_tokenizer(self):
        with open('model/history/_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer

    # load model
    def __myload_model(self):
        loaded_model = load_model('model/history/_lstm-model.h5')
        loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        self.model = loaded_model

    # text prediction
    def __kalimat(self, kalimat):
        # tokenization
        kalimat = self.tokenizer.texts_to_sequences(kalimat)
        kalimat = pad_sequences(kalimat, maxlen=119, dtype='int32', value=0)
        # prediction
        probability = self.model.predict(kalimat, batch_size=1, verbose = 0)[0]
        probability[0] = round(probability[0],2)*100
        probability[1] = round(probability[1],2)*100
        if(probability[0] < probability[1]):
            sentiment = "Hate Speech"
        else:
            sentiment = "Non Hate Speech"
        # prediction finished
        return sentiment, probability

    # tweets prediction
    def __tweets(self, tweets):
        # tokenization
        tweets = self.tokenizer.texts_to_sequences(tweets)
        tweets = pad_sequences(tweets, maxlen=119, dtype='int32', value=0)
        # prediction
        prediction = self.model.predict(tweets, batch_size=32, verbose = 0)
        sentiment_count = [0,0]
        encode_label = []
        for i in range(len(prediction)):
            if(prediction[i][0] < prediction[i][1]):
                encode_label.append("Hate Speech")
                sentiment_count[1]+=1
            else:
                encode_label.append("Non Hate Speech")
                sentiment_count[0]+=1
        prediction = np.array(encode_label)
        # prediction finished
        return prediction, sentiment_count

    # public, text prediction
    def predict_text(self, kalimat):
        sentiment, probability = self.__kalimat(kalimat)
        keras.backend.clear_session()
        print(colored("Text prediction finished", "green"))
        return sentiment, probability

    # public, tweets prediction
    def predict_tweets(self, tweets):
        prediction, sentiment_count = self.__tweets(tweets)
        keras.backend.clear_session()
        print(colored("Tweets prediction finished", "green"))
        return prediction, sentiment_count