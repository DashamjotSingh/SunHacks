import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

# Data overview
print(trainData.head())
print(trainData.info())

# Handling missing data: Impute missing values with the median
imputer = SimpleImputer(strategy='median')
trainData_filled = pd.DataFrame(imputer.fit_transform(trainData), columns=trainData.columns)

# Encoding categorical variables into numeric values (for both train and test sets)
trainData_encoded = pd.get_dummies(trainData_filled, drop_first=True)

# Feature scaling: Standardizing the features for linear regression
scaler = StandardScaler()
X = trainData_encoded.drop('price', axis=1)  # Features (all columns except 'price')
X_scaled = scaler.fit_transform(X)  # Scale features

y = trainData_encoded['price']  # Target (Price)

# Train-test split (reduced test size for better training)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 2. Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions on the test data
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluating the models
# Linear Regression
print("Linear Regression Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, lr_predictions))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, lr_predictions)))

# Random Forest
print("\nRandom Forest Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, rf_predictions))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, rf_predictions)))

# K-Fold Cross-Validation (optional for better model evaluation)
cv_scores_lr = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("\nLinear Regression CV RMSE:", np.sqrt(-cv_scores_lr.mean()))
print("Random Forest CV RMSE:", np.sqrt(-cv_scores_rf.mean()))

# Visualizing the results: Actual vs. Predicted for both models
plt.figure(figsize=(10,6))
plt.scatter(y_test, lr_predictions, label='Linear Regression', color='blue', alpha=0.6)
plt.scatter(y_test, rf_predictions, label='Random Forest', color='green', alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.show()
