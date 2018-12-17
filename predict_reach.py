"""
Method: supervised learning / linear regression.

Dataset: CSV exported from Facebook Ads Reporting.

Criterion variable:
- Budget

Predictor variable:
- Reach

Problem: if I spend this $X amount of money for this age group,
how many people can I reach via my ads?

"""
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# load dataset
dataset = pd.DataFrame.from_csv("fb_ads_2018.csv")
dg = dataset.groupby('Age').max()
print(dg)
print("=================================================")

# data cleaning
dataset = dataset.fillna(0)

dataset.plot(y='Amount Spent (USD)', x='Reach')
plot.title('Spend vs. Reach')
plot.ylabel('Amount Spent (USD)')
plot.xlabel('Reach')
plot.show()

# attributes and dependent variables
X = dataset[['Amount Spent (USD)']]
y = dataset['Reach']

# training and test dataset (30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# training algorithm
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# see attributes coefficients
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
print("=================================================")

# predict test dataset!
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual Reach': y_test, 'Predicted Reach': y_pred})
print(df)
print("=================================================")

# group by age
g = df.groupby('Age').mean()
print(g)

print("=================================================")

# predict new dataset
new_entry = {
    '25-34': "35"
}
new_dataset = pd.DataFrame(new_entry, index=[0])
new_pred = regressor.predict(new_dataset)
# correction of 20%
predicted_reach = int(new_pred[0])
print("Prediction: 25-34    35      {}".format(predicted_reach))
print("=================================================")

# evauate algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("=================================================")

# accuracy score
score = regressor.score(X_test, y_test)
print("Accuracy score: {}%".format(round(score * 100)))

print("=================================================")

# prediction graph
plot.scatter(X_train, y_train, color='red')
plot.plot(X_train, regressor.predict(X_train), color='blue')
plot.title('Age vs. Reach')
plot.xlabel('Age')
plot.ylabel('Reach')
plot.show()
