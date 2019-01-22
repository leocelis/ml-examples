import matplotlib.pyplot as plot
import numpy
import pandas as pd
from numpy import corrcoef

# load dataset
columns = ['Clicks (All)', 'Amount Spent (USD)', 'Reach', 'Results', 'Impressions', 'Frequency']
dataset = pd.read_csv("covariance.csv", index_col=None, usecols=columns)

# correlation coeficient
Sigma = corrcoef(dataset['Clicks (All)'], dataset.Reach)[1, 0]
print("=============================================================")
print("Correlation between Reach and Clicks is: {}".format(Sigma))
print("=============================================================")
print()
print()

# correlation matrix
fig, ax = plot.subplots()
corr = dataset.corr()
print("=============================================================")
print("Correlation Matrix")
print(corr)
print("=============================================================")
print()
print()

# chart
ax.matshow(corr, cmap='seismic')
# show values in the matrix
for (i, j), z in numpy.ndenumerate(corr):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
plot.xticks(range(len(dataset.columns)), dataset.columns)
plot.yticks(range(len(dataset.columns)), dataset.columns)
plot.show()
