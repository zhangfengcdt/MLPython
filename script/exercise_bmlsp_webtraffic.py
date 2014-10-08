"""
 Exercise 1 - Book<BMLSP>
 Tiny Machine Learning Example
"""
# Author: Feng Zhang <fzhang@esri.com>
# License: Simplified BSD

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Read the data from file
# Below are the first 3 lines
# "1	2272"
# "2	nan"
# "3	1386"
data_webtraffic = sp.genfromtxt("C:\\feng\\learn_python\\1400OS_Code\\1400OS_01_Codes\\data\\web_traffic.tsv", delimiter="\t")

# Clean and load the data
webTraffic_x = data_webtraffic[:,0];
webTraffic_y = data_webtraffic[:,1];
webTraffic_x = webTraffic_x[~sp.isnan(webTraffic_y)]
webTraffic_y = webTraffic_y[~sp.isnan(webTraffic_y)]

# Print and explore the data
plt.scatter(webTraffic_x, webTraffic_y)
plt.title("Web Traffice over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/Hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
# plt.show()

# Fit the polynomial regression model
fp1, residuals, rank, sv, rcond = sp.polyfit(webTraffic_x, webTraffic_y, 1, full=True)
fp1_fit = sp.poly1d(fp1)
# print(sp.sum( ((f_fit(webTraffic_x) - webTraffic_x))**2 ))
fpd, residuals, rank, sv, rcond = sp.polyfit(webTraffic_x, webTraffic_y, 3, full=True)
fpd_fit = sp.poly1d(fpd)

# Plot the fitted model
fx = sp.linspace(0, np.max(webTraffic_x), 1000)
plt.plot(fx, fp1_fit(fx), linewidth=4)
plt.legend(["d=%i" %fp1_fit.order], loc = "upper left")
plt.plot(fx, fpd_fit(fx), linewidth=4)
plt.legend(["d=%i" %fpd_fit.order], loc = "upper left")

plt.show()


