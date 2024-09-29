import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


data: DataFrame = pd.read_csv('housing.csv')

print(data.head())
print(data.info())

# Fill missing values
data = data.dropna()

# encoding variables into numeric values
data = pd.get_dummies(data, drop_first=True)

#x - test data, y - to be predicted
X = data.drop('price', axis=1)
y = data['price']  # Target

#model training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

#linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#predictions on the test data
lr_predictions = lr_model.predict(X_test)

#Linear Regression
print("Linear Regression Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, lr_predictions))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, lr_predictions)))

#Visualizing the results

plt.figure(figsize=(10,6))
plt.scatter(y_test, lr_predictions, label='Linear Regression', color='blue', alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.show()




