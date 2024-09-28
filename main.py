import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('housing.csv')

print(data.head())
print(data.info())

# 4. Data Preprocessing (Handling missing values, feature engineering)
# Fill missing values (if necessary) with median or mode or drop rows with missing data
data = data.dropna()

# encoding variables into numeric values
data = pd.get_dummies(data, drop_first=True)

#x - test data, y - to be predicted
X = data.drop('price', axis=1)  # Features (all columns except 'Price')
y = data['price']  # Target (Price)

# train and test the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
#linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)



# 8. Making predictions on the test data
lr_predictions = lr_model.predict(X_test)

# 9. Evaluating the model
# For Linear Regression
print("Linear Regression Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, lr_predictions))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, lr_predictions)))

# For Random Forest
print("\nRandom Forest Performance:")


# 10. Visualizing the results: Actual vs. Predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, lr_predictions, label='Linear Regression', color='blue', alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.show()
