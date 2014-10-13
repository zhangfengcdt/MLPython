"""
   Visualizing High Dimentional Data 
   Matplotlib.pyplot and Pandas packages
"""

# Author: Feng Zhang <fzhang@esri.com>
# License: Simplifided BSD

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import scatter_matrix


# Load dataset from the sklearn
iris_data = load_iris()

# Concantenate dataset to dataframe
iris_cat = np.concatenate((iris_data.data, iris_data.target.reshape(150,1)), axis=1)
iris_df = DataFrame(iris_cat, columns=['PA', 'PB', 'PC', 'PD','Name'])

# Plot the data using
# 1 - Parallel Coordinates
plt.figure()
parallel_coordinates(iris_df, 'Name')

# 2 - Andrews Curves
plt.figure()
andrews_curves(iris_df, 'Name')

# 3 - Scatter_Plots
plt.figure()
scatter_matrix(iris_df, alpha=0.2, figure=(6,6), diagonal='kde')

# Show the plot
plt.show()

