# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the Standard Libraries
2. Set Variables for Assigning Dataset Values
3. Import Linear Regression from sklearn
4. Assign Points for Representing in the Graph (for linear regression)
5. Predict the Regression for the Marks by Using the Representation of the Graph (Polynomial Regression)
6. Compare the Graphs and Conclude

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: SOUNDARYA J
RegisterNumber: 212223220108
*/
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv"
data = pd.read_csv(url)

# Display first few rows
print(data.head())

# Select relevant features and target variable
X = data[['enginesize']]  # Predictor
y = data['price']         # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Linear Regression ----
# Initialize and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions using the linear regression model
y_pred_linear = linear_model.predict(X_test)

# Evaluate the linear regression model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression MSE:", mse_linear)
print("Linear Regression R^2 score:", r2_linear)

# ---- Polynomial Regression ----
# Transform the features for Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize and train the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Make predictions using the polynomial regression model
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate the polynomial regression model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("Polynomial Regression MSE:", mse_poly)
print("Polynomial Regression R^2 score:", r2_poly)

# ---- Visualization ----
# Plot the results for linear regression
plt.scatter(X_test, y_test, color='red', label='Actual Prices')
plt.plot(X_test, y_pred_linear, color='blue', label='Linear Regression')
plt.title('Linear Regression for Predicting Car Prices')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot the results for polynomial regression
plt.scatter(X_test, y_test, color='red', label='Actual Prices')
plt.plot(X_test, y_pred_poly, color='green', label='Polynomial Regression')
plt.title('Polynomial Regression for Predicting Car Prices')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/935c2ced-4012-4969-9729-7734022b48c3)

![image](https://github.com/user-attachments/assets/80117b95-1783-4a48-8e4e-d207a531e93e)

![image](https://github.com/user-attachments/assets/7fea7daf-9527-45c3-ad8f-29405d72d53b)

## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
