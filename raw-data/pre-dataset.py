import numpy as np
import pandas as pd

import re
import sys


data = pd.read_csv('data-hs/re_dataset.csv', encoding='latin-1')
data = data.head(20)

columns = ['Abusive','HS_Individual','HS_Group','HS_Religion','HS_Race','HS_Physical','HS_Gender','HS_Other','HS_Weak','HS_Moderate','HS_Strong']
data = data.drop(columns, axis='columns')
data = data.rename(columns={'Tweet': 'text', 'HS': 'sentiment'})

alay_dict = pd.read_csv('data-hs/new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 
                                      1: 'replacement'})

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('user',' ',text) # Remove every username
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    
    return text
    
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text

alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

#def remove_stopword(text):
#    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
#    text = re.sub('  +', ' ', text) # Remove extra spaces
#    text = text.strip()
#    return text

def stemming(text):
    return stemmer.stem(text)

def preprocess(text):

    #lowercase
    text = lowercase(text) # 1
    #remove_nonaplhanumeric
    text = remove_nonaplhanumeric(text) # 2
    #remove_unnecessary_char
    text = remove_unnecessary_char(text) # 2
    #normalize_alay
    text = normalize_alay(text) # 3
    #stemming
    text = stemming(text) # 4
    #remove_stopword
    #text = remove_stopword(text) # 5
    return text

data['text'] = data['text'].apply(preprocess)

data.to_csv('preprocessed_dataset.csv', index=False)
data.to_excel('preprocessed_dataset.xlsx')

