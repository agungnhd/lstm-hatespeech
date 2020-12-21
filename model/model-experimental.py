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
        print(colored("Model loaded", "green"))

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
        units = 64

        # Embedding Layer
        embedding_W = self.model.layers[0].get_weights()[0]

        # LSTM Layer
        lstm_W = self.model.layers[2].get_weights()[0]
        lstm_U = self.model.layers[2].get_weights()[1]
        lstm_b = self.model.layers[2].get_weights()[2]
        # W per Gate
        W_i = lstm_W[:, :units]
        W_f = lstm_W[:, units:units*2]
        W_c = lstm_W[:, units*2:units*3]
        W_o = lstm_W[:, units*3:]
        # U per Gate
        U_i = lstm_W[:, :units]
        U_f = lstm_W[:, units:units*2]
        U_c = lstm_W[:, units*2:units*3]
        U_o = lstm_W[:, units*3:]
        # b per Gate
        b_i = lstm_W[:, :units]
        b_f = lstm_W[:, units:units*2]
        b_c = lstm_W[:, units*2:units*3]
        b_o = lstm_W[:, units*3:]

        embedding_W = pd.DataFrame(data = embedding_W)
        print(embedding_W)
        embedding_W.to_excel('model/raw_1/1-embedding_W.xlsx')

        lstm_W = pd.DataFrame(data = lstm_W)
        print(lstm_W)
        lstm_U = pd.DataFrame(data = lstm_U)
        print(lstm_U)
        lstm_b = pd.DataFrame(data = lstm_b)
        print(lstm_b)

        lstm_W.to_excel('model/raw_1/1-lstm_W.xlsx')
        lstm_U.to_excel('model/raw_1/1-lstm_U.xlsx')
        lstm_b.to_excel('model/raw_1/1-lstm_b.xlsx')
        

        W_i = pd.DataFrame(data = W_i)
        W_f = pd.DataFrame(data = W_f)
        W_c = pd.DataFrame(data = W_c)
        W_o = pd.DataFrame(data = W_o)

        U_i = pd.DataFrame(data = U_i)
        U_f = pd.DataFrame(data = U_f)
        U_c = pd.DataFrame(data = U_c)
        U_o = pd.DataFrame(data = U_o)
        
        b_i = pd.DataFrame(data = b_i)
        b_f = pd.DataFrame(data = b_f)
        b_c = pd.DataFrame(data = b_c)
        b_o = pd.DataFrame(data = b_o)


        W_i.to_excel('model/raw_1/2-lstm_W_i.xlsx')
        W_f.to_excel('model/raw_1/2-lstm_W_f.xlsx')
        W_c.to_excel('model/raw_1/2-lstm_W_c.xlsx')
        W_o.to_excel('model/raw_1/2-lstm_W_o.xlsx')

        U_i.to_excel('model/raw_1/2-lstm_U_i.xlsx')
        U_f.to_excel('model/raw_1/2-lstm_U_f.xlsx')
        U_c.to_excel('model/raw_1/2-lstm_U_c.xlsx')
        U_o.to_excel('model/raw_1/2-lstm_U_o.xlsx')

        b_i.to_excel('model/raw_1/2-lstm_b_i.xlsx')
        b_f.to_excel('model/raw_1/2-lstm_b_f.xlsx')
        b_c.to_excel('model/raw_1/2-lstm_b_c.xlsx')
        b_o.to_excel('model/raw_1/2-lstm_b_o.xlsx')


    def print_tokenizer(self):
        word_index = self.tokenizer.word_index
        word_index = pd.DataFrame(list(word_index.items()), columns = ['text','index'])
        print(word_index)
        word_index.to_excel('model/raw_1/0-word_index.xlsx')
        

    def main(self):
        self.print_tokenizer()
        self.print_weight()


experiment = model_experimental()
experiment.main()