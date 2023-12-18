import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x_values = np.array([64, 75, 68, 73, 78, 82, 76, 85, 71, 88]).reshape(-1, 1)
y_values = np.array([17, 27, 15, 24, 39, 34, 30, 48, 19, 47])

lr = LinearRegression()
lr.fit(x_values, y_values)

slope = lr.coef_[0]
intercept = lr.intercept_

print("Slope :", slope)
print("intercept :", intercept)
