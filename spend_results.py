"""
Predict Results for a given spend
"""

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load dataset
dataset = pd.DataFrame.from_csv("spend_results.csv", index_col=None)

# data cleaning
dataset = dataset.fillna(0)
dataset = dataset[dataset['Amount Spent (USD)'] > 0]

# feature
X = dataset[['Amount Spent (USD)']]

# target
y = dataset['Results']

# training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)

# training algorithm
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# attributes coefficients
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print("=================================================")
print(coeff_df)
print("=================================================")
print()
print()

# predict test dataset!
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual Results': y_test, 'Predicted Results': y_pred})
# print("=================================================")
# print("Test dataset predictions")
# print(df)
# print("=================================================")
# print()
# print()

# predict new dataset
new_dataset = pd.DataFrame.from_csv("spend_results_predict.csv", index_col=None)
new_dataset = new_dataset.fillna(0)
new_dataset = new_dataset[new_dataset['Amount Spent (USD)'] > 0]
new_pred = regressor.predict(new_dataset)
new_dataset['Predicted Results'] = new_pred
print("=================================================")
print("New dataset predictions")
print(new_dataset)
print("=================================================")
print()
print()

# evaluate algorithm
print("=================================================")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("=================================================")
print()
print()

# accuracy score
score = regressor.score(X_test, y_test)
print("=================================================")
print("Accuracy score: {}%".format(round(score * 100)))
print("=================================================")
print()
print()

# prediction graph
plot.scatter(X_train, y_train, color='red')
plot.plot(X_train, regressor.predict(X_train), color='blue')
plot.title('Spend vs. Results')
plot.xlabel('Amount Spent (USD)')
plot.ylabel('Results')
plot.show()
# plot.savefig('spend_results.png')
