import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv("Salary_Data.csv")

# Prepare the data
x = data['YearsExperience'].values.reshape(-1, 1)
y = data['Salary'].values

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# Make predictions on the test set
predictions = linear_model.predict(x_test)

# Evaluate the model
r2score = r2_score(y_test, predictions)
print("R2_Score:", r2score)

# Plot the data points and regression line
plt.scatter(x_test, y_test, c="yellow", label="Data Points")
plt.plot(x_test, predictions, c="blue", linewidth=3, label="Regression Line")
plt.show()
