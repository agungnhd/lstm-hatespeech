import re
import nltk
import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#nltk.download('stopwords')
from nltk.corpus import stopwords
list_stopwords = set(stopwords.words('indonesian'))

from termcolor import colored


class preprocessing:

    def __init__(self):
        self.initialize_stemmer()
        self.initialize_slangwords_dictionary()

    def initialize_stemmer(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
    
    def initialize_slangwords_dictionary(self):
        slang_dict = pd.read_csv('data/dictionary/kamus_slangwords.csv', encoding='latin-1', header=None)
        slang_dict = slang_dict.rename(columns={0: 'original', 1: 'replacement'})
        self.slang_dict_map = dict(zip(slang_dict['original'], slang_dict['replacement']))

    # remove newline ('\n')
    def remove_newline(self, text):
        return re.sub('\n',' ',text)

    # text cleaning
    def text_cleaning(self, text):
        text = re.sub(r'@[^\s]+',' ',text) # removing @username
        text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+)|(pic.twitter.com/[^\s]+))',' ',text) # removing URL
        text = text.lower() # case folding
        text = re.sub('[^0-9a-zA-Z]+', ' ', text) # removing non alphabet & numeric
        text = re.sub(r'(^| ).( |$)',' ',text) # removing single characters
        text = re.sub('  +', ' ', text) # removing extra spaces
        text = text.strip() # stripping

        return text

    # stemming
    def stemming(self, text):
        return self.stemmer.stem(text)

    def normalization(self, text):
        return ' '.join([self.slang_dict_map[word] if word in self.slang_dict_map else word for word in text.split(' ')])

    def remove_stopword(self, text):
        text = ' '.join(['' if word in list_stopwords else word for word in text.split(' ')])
        text = re.sub('  +', ' ', text) # Remove extra spaces
        text = text.strip()
        return text
    
    # text preprocessing
    def text_preprocessing(self, text):
        text = self.remove_newline(text)
        text = self.text_cleaning(text)
        text = self.normalization(text)
        text = self.stemming(text)
        text = self.remove_stopword(text)
        # preprocessing finished
        return text
    
    # public, text preprocessing
    def clean_text(self, data):
        data = pd.DataFrame([data], columns = ['text']) # create single row dataframe
        data['text'] = data['text'].apply(self.remove_newline)
        data['preprocessed_text'] = data['text'].apply(self.text_preprocessing)
        # text preprocessed
        print(colored("Text preprocessed", "green"))
        return data
    
    # public, tweets preprocessing
    def clean_tweet(self, data):
        data['text'] = data['text'].apply(self.remove_newline)
        data['preprocessed_text'] = data['text'].apply(self.text_preprocessing)
        # tweets preprocessed
        print(colored("Tweets preprocessed", "green"))
        return data

    # public, dataset preprocessing
    def clean_dataset(self, data):
        data['text'] = data['text'].apply(self.remove_newline)
        data['text'] = data['text'].apply(self.text_preprocessing)
        # dataset preprocessed
        print(colored("Dataset preprocessed", "green"))
        return data