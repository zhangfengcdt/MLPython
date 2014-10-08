"""

Template to do Regression

"""
# Author: Feng Zhang <fzhang@esri.com>
# License: Simplified BSD

import sys
from sklearn import linear_model
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

###################################################

# 3 - define the regression model 
clf = linear_model.LinearRegression()

###################################################

# 4 - fit the model
clf.fit (data_train, y_train)
clf.coef_

# 5 - predict using the model
y_predicted = clf.predict(y_train)

# 6 - validata the model
print(metrics.explained_variance_score(y_test, y_predicted))  #ES = SSR/SST
print(metrics.mean_absolute_error(y_test, y_predicted))       #MAE (l1)
print(metrics.mean_square_error(y_test, y_predicted))         #MSE (l2)
print(metrics.r2_score(y_test, y_predicted))                  #R2 = 1-SSE/SST

# 7 - print the result
import matplotlib.pyplot as plt
plt.scatter(data_test, y_test,  color='black')
plt.plot(data_test, y_predicted), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
