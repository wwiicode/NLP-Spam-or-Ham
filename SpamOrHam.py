#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 22:01:40 2018

@author: zhihuanwilson
"""

nltk.download()

import nltk
import string
import re
import pandas as pd
pd.set_option('display.max_colwidth', 100)

wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()



rawData = open("SMSSpamCollection.tsv").read()

rawData[0:500]

parsedData = rawData.replace('\t', '\n').split('\n')

parsedData[0:5]

labelList = parsedData[0::2]
textList = parsedData[1::2]

fullCorpus = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
fullCorpus.columns = ('label', 'body_text')
fullCorpus.head()

### Explore the dataset
# What is the shape of the dataset?
print("Input data has {} rows and {} columns".format(len(fullCorpus), len(fullCorpus.columns)))

# How many spam/ham are there?
print("Out of {} rows, {} are spam, {} are ham".format(len(fullCorpus),
                                                       len(fullCorpus[fullCorpus('label')=='spam']),
                                                       len(fullCorpus[fullCorpus('label')=='ham'])))

# How much missing date is there?
print("Number of null in label: {}".format(fullCorpus['label'].isnull().sum()))
print("Number of null in text: {}".format(fullCorpus['body_text'].isnull().sum()))



stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

# Create function to remove punctuation, tokenize, remove stopwords, and stem
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text 



from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer=clean_text)
X_counts = count_vect.fit_transform(data['body_text'])
print(X_counts.shape)
print(count_vect.get_feature_names())

data_sample = data[0:20]
count_vect_sample = CountVectorizer(analyzer=clean_text)
X_counts_sample = count_vect_sample.fit_transform(data_sample['body_text'])
print(X_counts_sample.shape)
print(count_vect_sample.get_feature_names())

#Vectorizer output sparse matrices
X_counts_sample

X_counts_df = pd.DataFrame(X_counts_sample.toarray())
X_counts_df

X_counts_df.columns = count_vect_sample.get_feature_names()
X_counts_df

########################################################## TFIDF #######
## Inverse document frequency weighting
## Apply TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())




# take a sample from the data
data_sample = data[0:20]
tfidf_vect_sample = TfidfVectorizer(analyzer=clean_text)
X_tfidf_sample = tfidf_vect_sample.fit_transform(data_sample['body_text'])
print(X_tfidf_sample.shape)
print(tfidf_vect.get_feature_names())

# Vectorizers output sparse matrices

X_tfidf_df = pd.DataFrame(X_tfidf_sample.toarray())
X_tfidf_df.columns = tfidf_vect_sample.get_feature_names()
X_tfidf_df


## Feature Engineering
# Feature Creation for text message length
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data.head()

# Create feature for % of text that is punctuation
# reload the data so we have full text
data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']


def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))
data.head()

# Evaluate created feature
from matplotlib import pyplot
import numpy as np

bins = np.linspace(0, 200, 40)
pyplot.hist(data[data['label']=='spam']['body_len%'], bins, alpha=0.5, normed=True, label='spam')
pyplot.hist(data[data['label']=='ham']['body_len%'], bins, alpha=0.5, normed=True, label='ham')
pyplot.legend(loc='upper left')

bins = np.linspace(0, 200, 40)
pyplot.hist(data[data['label']=='spam']['punct%'], bins, alpha=0.5, normed=True, label='spam')
pyplot.hist(data[data['label']=='ham']['punct%'], bins, alpha=0.5, normed=True, label='ham')
pyplot.legend(loc='upper right')


## Building a Random Forest Model
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])

X_features = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)









