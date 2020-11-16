import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import pandas as pd
import numpy as np
import pickle

import time

import matplotlib.pyplot as plt
import seaborn as sns

from termcolor import colored


class lstm_final:

    def __init__(self):
        #Base_Path Configuration
        #self.BASE_PATH = '/content/gdrive/My Drive/agungnhd-lstmsentiment/'
        self.BASE_PATH = ''

        keras.backend.clear_session()

        self.timestr = time.strftime("%Y%m%d-%H%M%S")


    # Load data
    def __load_dataset(self):
        self.train_data = pd.read_excel('{}data/preprocessed_dataset.xlsx'.format(self.BASE_PATH))
        print(colored("Dataset loaded", "green"))

    # Tokenization
    def __tokenization(self):
        train_data = self.train_data

        tokenizer = Tokenizer(num_words = 2000, split = ' ')
        tokenizer.fit_on_texts(train_data['text'].astype(str).values)
        tokenized_data = tokenizer.texts_to_sequences(train_data['text'].astype(str).values)
        #max_len = max([len(i) for i in tokenized_data])
        tokenized_data = pad_sequences(tokenized_data, maxlen = 64)
        print(colored("Tokenizing and padding completed", "green"))

        with open('{}model/history/_tokenizer.pickle'.format(self.BASE_PATH), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('{0}model/history/tokenizer-{1}.pickle'.format(self.BASE_PATH, self.timestr), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(colored("Tokenizer saved", "green"))

        self.tokenized_data = tokenized_data

    # Building the model
    def __build_model(self):

        #data_shape = self.tokenized_data.shape[1]

        print(colored("Creating the LSTM model", "yellow"))
        model = Sequential()
        model.add(Embedding(2000, 32, input_length = 64))
        model.add(SpatialDropout1D(0.8))
        model.add(LSTM(32, dropout = 0.8))
        model.add(Dense(2, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.summary()

        return model

    # Training the model
    def __training_model(self, n_epoch):
        
        model = self.model
        tokenized_data = self.tokenized_data
        train_data = self.train_data

        print(colored("Training the LSTM model (main)", "yellow"))
        X_train = tokenized_data
        Y_train = pd.get_dummies(train_data['sentiment']).values

        history = model.fit(X_train, Y_train, epochs = n_epoch, batch_size = 256)
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
        print(colored("Starting K-Fold Cross Validation", "yellow"))

        tokenized_data = self.tokenized_data
        train_data = self.train_data

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # split into input (X) and output (Y) variables
        X = tokenized_data
        Y = train_data['sentiment']

        kfold = StratifiedKFold(n_splits=n_kfold, shuffle=True, random_state=seed)
        
        results = []
        train_val_history = []
        
        fold_count = 1
        for train, test in kfold.split(X, Y):
            print(colored("Fold-{}".format(fold_count), "green"))
            
            # Create model for each fold
            model = self.__build_model()

            # Fit the model
            history = model.fit(X[train], pd.get_dummies(Y[train]).values, epochs = n_epoch, batch_size = 256, validation_data=(X[test], pd.get_dummies(Y[test]).values))

            # Evaluate the model
            
            # Predict classes
            pred_class = model.predict_classes(X[test], batch_size=256)
            pred_class = pred_class.flatten()
            
            print(colored("Fold-{} Performance".format(fold_count), "green"))
            accuracy = accuracy_score(Y[test], pred_class) # accuracy: (tp + tn) / (p + n)
            print('Accuracy: %f' % accuracy)
            precision = precision_score(Y[test], pred_class) # precision: tp / (tp + fp)
            print('Precision: %f' % precision)
            recall = recall_score(Y[test], pred_class) # recall: tp / (tp + fn)
            print('Recall: %f' % recall)
            f1 = f1_score(Y[test], pred_class) # f1: 2 tp / (2 tp + fp + fn)
            print('F1 score: %f' % f1)
            
            train_val_history.append(pd.DataFrame(history.history))

            temp = [fold_count-1, accuracy*100, precision*100, recall*100, f1*100]
            results.append(temp)

            fold_count = fold_count+1

        report = pd.DataFrame(data=results, columns=["K", "accuracy", "precision", "recall", "f1"])
        
        df_concat = pd.concat(train_val_history)
        by_row_index = df_concat.groupby(df_concat.index)
        history_avg = by_row_index.mean()

        self.validation_history = history_avg

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
        history_df.insert(loc=0, column='epoch', value=history_df.index+1)
        history_df.to_excel("{0}data/result_history/_training_report.xlsx".format(self.BASE_PATH))
        history_df.to_excel("{0}data/result_history/training_report-{1}.xlsx".format(self.BASE_PATH, self.timestr))
        self.training_report = history_df
        print(colored("Training Report saved", "green"))
    
    # save training report to excel
    def __save_training_val_report(self):
        # save report
        history_df = pd.DataFrame(self.validation_history)
        history_df.insert(loc=0, column='epoch', value=history_df.index+1)
        history_df.to_excel("{0}data/result_history/_val_training_report.xlsx".format(self.BASE_PATH))
        history_df.to_excel("{0}data/result_history/val_training_report-{1}.xlsx".format(self.BASE_PATH, self.timestr))
        self.validation_training_report = history_df
        print(colored("Training Report saved", "green"))

    # save model and architecture to single file (.h5)
    def __save_model(self):
        self.model.save("{}model/history/_lstm-model.h5".format(self.BASE_PATH))
        self.model.save("{0}model/history/lstm-model-{1}.h5".format(self.BASE_PATH, self.timestr))
        print(colored("LSTM Model saved", "green"))


    def __validation_plot(self):
        #reshape dataframe
        df_sns = pd.melt(self.validation_report, id_vars="K", var_name="metrics", value_name="percentage")
        plt.figure()
        plt.style.use('ggplot')
        sns_fig = sns.catplot(x='K', y='percentage', hue='metrics', data=df_sns, kind='bar', palette="muted")
        sns_fig.savefig("{0}data/figure_history/_validation_report.png".format(self.BASE_PATH), dpi=600)
        sns_fig.savefig("{0}data/figure_history/validation_report-{1}.png".format(self.BASE_PATH, self.timestr), dpi=600)
        print(colored("Figure saved", "green"))

    def __training_plot(self):
        #reshape dataframe
        df_sns = pd.melt(self.training_report, id_vars="epoch", var_name="loss_acc", value_name="percentage")
        plt.figure()
        plt.style.use('ggplot')
        sns_fig = sns.lineplot(x='epoch', y='percentage', hue='loss_acc', data=df_sns, palette="muted")
        sns_fig.set(ylabel=None)
        sns_fig.figure.savefig("{0}data/figure_history/_training_report.png".format(self.BASE_PATH), dpi=600)
        sns_fig.figure.savefig("{0}data/figure_history/training_report-{1}.png".format(self.BASE_PATH, self.timestr), dpi=600)
        print(colored("Figure saved", "green"))

    def __validation_training_plot(self):
        #reshape dataframe
        df_sns = pd.melt(self.validation_training_report, id_vars="epoch", var_name="loss_acc_validation", value_name="percentage")
        plt.figure()
        plt.style.use('ggplot')
        sns_fig = sns.lineplot(x='epoch', y='percentage', hue='loss_acc_validation', data=df_sns, palette="muted")
        sns_fig.set(ylabel=None)
        sns_fig.figure.savefig("{0}data/figure_history/_validation_training_report.png".format(self.BASE_PATH), dpi=600)
        sns_fig.figure.savefig("{0}data/figure_history/validation_training_report-{1}.png".format(self.BASE_PATH, self.timestr), dpi=600)
        print(colored("Figure saved", "green"))


    def training(self, n_epoch):
        self.model = self.__build_model()
        self.__training_model(n_epoch)
        self.__save_model()
        self.__save_training_report()
        self.__training_plot()
    
    def validation(self, n_epoch, n_kfold):
        self.__kfold_validation(n_epoch, n_kfold)
        self.__save_validation_report()
        self.__validation_plot()

        self.__save_training_val_report()
        self.__validation_training_plot()


    def main(self, n_epoch, n_kfold):
        # data preparations
        self.__load_dataset()
        self.__tokenization()
        
        self.training(n_epoch) # build and save
        self.validation(n_epoch,n_kfold) # build and evaluate

        keras.backend.clear_session()

    def debug(self):
        print(self.BASE_PATH)

lstm = lstm_final()
lstm.main(50, 5)
# main(epoch, fold)
# best model :
# embedding 32 units, lstm 32 units, each with dropout 0.8, epochs 20