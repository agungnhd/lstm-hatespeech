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

input_array = np.random.randint(1000, size=(32, 10))

model = Sequential()
model.compile('rmsprop', 'mse')

output_array = model.predict(input_array)

print(output_array)