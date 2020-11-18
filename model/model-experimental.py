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


class model_experimental:

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

    def print_weight(self):
        print(colored("Experimental", "green"))
        w_1 = self.model.layers[0].get_weights()[0]
        w_3 = self.model.layers[2].get_weights()[0]

        df_w1 = pd.DataFrame(data = w_1)
        print(df_w1)
        df_w3 = pd.DataFrame(data = w_3)
        print(df_w3)

        df_w1.to_excel('model/weight_1.xlsx')
        df_w3.to_excel('model/weight_3.xlsx')

    def print_tokenizer(self):
        word_index = self.tokenizer.word_index
        word_index = pd.DataFrame(list(word_index.items()), columns = ['text','index'])
        print(word_index)

        word_index.to_excel('model/word_index.xlsx')
        

    def main(self):
        self.print_weight()
        self.print_tokenizer()


experiment = model_experimental()
experiment.main()