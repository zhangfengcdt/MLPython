"""

 Template to do Classification

"""
# Author: Feng Zhang <fzhang@esri.com>
# License: Simplified BSD

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# 1 - load data
data_folder = sys.argv[1]
dataset = load_files(data_folder, shuffle=False)
print("n_samples: %d" % len(dataset.data))

# 2 - split the dataset in training and test set:
data_train, data_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.25, random_state=None)

################MODEL################################

# 3 - define the classification model 
clf_lsvc = Pipline(['vect', TfidfVectorizer(min_df=0.95, max_df=3),
               'classifier', LinearSVC(C=1000) ])

#####################################################

# 3 - tune parameters
parameters = {'vect__ngram_rannge': [(1,1),(1,2)]}
gs_clf = GridSearchCV(clf_lsvc, parameters, n_jobs=-1)

# 4 - fit the model
gs_clf = gs_clf.fit(data_train, y_train)
print(gs_clf.scorer_)

# 5 - predict using the model
y_predicted = gs_clf.predict(docs_test)

# 6 - validate the model 
print(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names))
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)

# 7 - print the result
import matplotlib.pyplot as plt
plt.matshow(cm)
plt.show()
