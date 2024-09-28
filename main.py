# 1. Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the dataset (Replace 'your_dataset.csv' with the actual file path)
# Here, we assume the dataset has columns like 'Bedrooms', 'Bathrooms', 'SquareFootage', 'Price', etc.
# You can download a dataset like the one from Kaggle (House Prices - Advanced Regression Techniques)
data = pd.read_csv('your_dataset.csv')

# 3. Inspecting the dataset
print(data.head())
print(data.info())

# 4. Data Preprocessing (Handling missing values, feature engineering)
# Fill missing values (if necessary) with median or mode or drop rows with missing data
data = data.dropna()

# One-hot encoding for categorical variables like 'City', 'Neighborhood' (if present)
data = pd.get_dummies(data, drop_first=True)

# 5. Split the data into features (X) and target (y)
X = data.drop('Price', axis=1)  # Features (all columns except 'Price')
y = data['Price']  # Target (Price)

# 6. Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Building and training the model
# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model 2: Random Forest (optional if you want to try another model)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 8. Making predictions on the test data
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# 9. Evaluating the model
# For Linear Regression
print("Linear Regression Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, lr_predictions))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, lr_predictions)))

# For Random Forest
print("\nRandom Forest Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, rf_predictions))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, rf_predictions)))

# 10. Visualizing the results: Actual vs. Predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, lr_predictions, label='Linear Regression', color='blue', alpha=0.6)
plt.scatter(y_test, rf_predictions, label='Random Forest', color='green', alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.show()
