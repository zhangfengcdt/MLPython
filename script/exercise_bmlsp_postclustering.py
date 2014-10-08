"""
 Exercise 2 - Posts<BMLSP>
 Clustering Machine Learning Example
"""
# Author: Feng Zhang <fzhang@esri.com>
# License: Simplified BSD

import os
import sys
import numpy as np
import scipy as sp
import nltk.stem
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Define Stemmed Class for TfidfVectorizer
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

# Read the data from file
data_folder = sys.argv[1]
data_train = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=data_folder, categories=['comp.graphics', 'com.sys.ibm.pc.hardware', 'sci.space'])
data_test = sklearn.datasets.load_mlcomp("20news-18828", "test", mlcomp_root=data_folder, categories=['comp.graphics', 'com.sys.ibm.pc.hardware', 'sci.space'])
print(len(data_train.filenames))

# Clean and load the data
vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', charset_error='ignore')
data_train_vector = vectorizer.fit_tranform(data_train.data)
data_test_vector = vectorizer.tranform(data_test.data)

# Print and explore the data
num_samples, num_features = data_train_vector.shape
print("# of Samples %d, # of Features %d"  %(num_samples, num_features))

# Fit the polynomial regression model
clf_KMeans = KMeans(n_clusters=50, init="ramdom", n_init=1, verbose=1).fit(data_train_vector)

# Plot the fitted model
print(clf_KMeans.labels_)
print(clf_KMeans.labels_.shape)

# predict using the model
y_predicted = clf_KMeans.predict(data_test_vector)

# validate the model 
# print the result
