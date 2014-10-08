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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Define Stemmed Class for TfidfVectorizer
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

# Read the data from file
data_folder = sys.argv[1]
data_posts = [open(os.path.join(data_folder, f)).read() for f in os.listdir(data_folder)]

# Clean and load the data
vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', charset_error='ignore')
data_train_vector = vectorizer.fit_transform(data_posts)
num_train_samples, num_train_features = data_train_vector.shape
data_test_post = "imaging databases"
data_test_vector = vectorizer.transform(data_test_post)

# Print and explore the data
print("# of Samples %d, # of Features %d" %(num_train_samples, num_train_features))
print(vectorizer.get_feature_names())
for k in range(num_train_samples):
	print(data_train_vector.getrow(3).toarray())
print(vectorizer.get_feature_names())
print(data_test_vector)

# Fit the polynomial regression model


# Plot the fitted model

