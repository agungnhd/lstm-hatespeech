import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import pandas as pd

import numpy as np

import pickle

import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

from termcolor import colored


class lstm_final:

    def __init__(self):
        #Google Colab self.Base_Path
        #self.BASE_PATH = '/content/gdrive/My Drive/agungnhd-lstmsentiment/'
        self.BASE_PATH = ''

        keras.backend.clear_session()

        self.timestr = time.strftime("%Y%m%d-%H%M%S")


    # Load data
    def __load_dataset(self):
        print(colored("Loading train and test data", "yellow"))
        self.train_data = pd.read_excel('{}data/preprocessed_dataset.xlsx'.format(self.BASE_PATH))

    # Tokenization
    def __tokenization(self):
        train_data = self.train_data

        print(colored("Tokenizing and padding data", "yellow"))
        tokenizer = Tokenizer(num_words = 2000, split = ' ')
        tokenizer.fit_on_texts(train_data['text'].astype(str).values)
        tokenized_data = tokenizer.texts_to_sequences(train_data['text'].astype(str).values)
        max_len = max([len(i) for i in tokenized_data])
        tokenized_data = pad_sequences(tokenized_data, maxlen = max_len)
        
        print(colored("Tokenizing and padding completed", "green"))

        with open('{}model/history/_tokenizer.pickle'.format(self.BASE_PATH), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('{0}model/history/tokenizer-{1}.pickle'.format(self.BASE_PATH, self.timestr), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(colored("Tokenizer saved", "green"))

        self.tokenized_data = tokenized_data

    # Building the model
    def __build_model(self):

        tokenized_data = self.tokenized_data

        print(colored("Creating the LSTM model", "yellow"))
        model = Sequential()
        model.add(Embedding(2000, 128, input_length = tokenized_data.shape[1]))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(256, dropout = 0.2))
        model.add(Dense(2, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.summary()

        self.model = model

    # Training the model
    def __training_model(self, n_epoch):
        
        model = self.model
        tokenized_data = self.tokenized_data
        train_data = self.train_data

        print(colored("Training the LSTM model", "green"))
        X_train = tokenized_data
        Y_train = pd.get_dummies(train_data['sentiment']).values

        history = model.fit(X_train, Y_train, epochs = n_epoch, batch_size = 128)
        print(colored(history, "green"))

        with open('{}model/history/_trainHistoryDict'.format(self.BASE_PATH), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        with open('{0}model/history/trainHistoryDict-{1}'.format(self.BASE_PATH, self.timestr), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        
        print(colored("Training History saved", "green"))
        self.history = history
        self.model = model
    
    # kfold validation
    def __kfold_validation(self, n_epoch, n_kfold):

        tokenized_data = self.tokenized_data
        train_data = self.train_data

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # split into input (X) and output (Y) variables
        X = tokenized_data
        Y = train_data['sentiment']

        # define 10-fold cross validation test harness
        kfold = StratifiedKFold(n_splits=n_kfold, shuffle=True, random_state=seed)
        
        results = []
        
        fold_count = 1
        for train, test in kfold.split(X, Y):
            print(colored("Fold-{}".format(fold_count), "green"))
            fold_count = fold_count+1
            
            #Create model
            model = Sequential()
            model.add(Embedding(2000, 128, input_length = X.shape[1]))
            model.add(SpatialDropout1D(0.4))
            model.add(LSTM(256, dropout = 0.2))
            model.add(Dense(2, activation = 'sigmoid'))

            # Compile model
            model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            #model.summary()

            # Fit the model
            model.fit(X[train], pd.get_dummies(Y[train]).values, epochs = n_epoch, batch_size = 128)

            # evaluate the model
            
            # Predict classes
            pred_class = model.predict_classes(X[test], batch_size=128)
            pred_class = pred_class.flatten()
            
            # accuracy: (tp + tn) / (p + n)
            accuracy = accuracy_score(Y[test], pred_class)
            print('Accuracy: %f' % accuracy)
            # precision: tp / (tp + fp)
            precision = precision_score(Y[test], pred_class)
            print('Precision: %f' % precision)
            # recall: tp / (tp + fn)
            recall = recall_score(Y[test], pred_class)
            print('Recall: %f' % recall)
            # f1: 2 tp / (2 tp + fp + fn)
            f1 = f1_score(Y[test], pred_class)
            print('F1 score: %f' % f1)
            
            temp = [fold_count-1, accuracy*100, precision*100, recall*100, f1*100]
            results.append(temp)

        report = pd.DataFrame(data=results, columns=["K", "accuracy", "precision", "recall", "f1"])
        
        self.validation_report = report

    # save validation report to excel
    def __save_validation_report(self):
        # save report
        self.validation_report.to_excel("{0}data/result_history/_validation_report.xlsx".format(self.BASE_PATH))
        self.validation_report.to_excel("{0}data/result_history/validation_report-{1}.xlsx".format(self.BASE_PATH, self.timestr))
        print(colored("Validation Report saved", "green"))
    
    # save training report to excel
    def __save_training_report(self):
        # save report
        history_df = pd.DataFrame(self.history.history)
        history_df['loss'] = history_df['loss']*100
        history_df['accuracy'] = history_df['accuracy']*100
        history_df.insert(loc=0, column='epoch', value=history_df.index+1)
        history_df.to_excel("{0}data/result_history/_training_report.xlsx".format(self.BASE_PATH))
        history_df.to_excel("{0}data/result_history/training_report-{1}.xlsx".format(self.BASE_PATH, self.timestr))
        print(colored("Training Report saved", "green"))

    # save model and architecture to single file (.h5)
    def __save_model(self):
        self.model.save("{}model/history/_lstm-model.h5".format(self.BASE_PATH))
        self.model.save("{0}model/history/lstm-model-{1}.h5".format(self.BASE_PATH, self.timestr))
        print(colored("LSTM Model saved", "green"))

    # training lstm model
    def main(self, n_epoch):
        self.__load_dataset()
        self.__tokenization()
        self.__build_model()
        self.__training_model(n_epoch)
        self.__save_model()
        self.__save_training_report()

        keras.backend.clear_session()
    
    # model evaluation only
    def validation(self, n_epoch, n_kfold):
        self.__load_dataset()
        self.__tokenization()
        self.__kfold_validation(n_epoch, n_kfold)
        self.__save_validation_report()

        keras.backend.clear_session()

    def debug(self):
        print(self.BASE_PATH)


#
# 
#
# main(jumlah_epoch) -> build and save
# validation(jumlah_epoch, jumlah_kfold) -> build and evaluate
