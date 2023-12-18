import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ch = fetch_california_housing()
df = pd.DataFrame(data=ch.data,columns=ch.feature_names)
df['target'] = ch.target

# Display the first few rows of the dataset
print("Dataset Head:")
pd.set_option('display.max_columns', None)
print(df)


x = df.drop('target',axis=1)
y = df['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)

result = mean_squared_error(y_test,pred)

print("Mean Squared Error ::",result)

new_data = pd.DataFrame({
    'MedInc': [5.0, 6.0],
    'HouseAge': [20, 25],
    'AveRooms': [6.0, 7.0],
    'AveBedrms': [1.0, 1.2],
    'Population': [1500, 1800],
    'AveOccup': [3.0, 3.5],
    'Latitude': [34.0, 35.0],
    'Longitude': [-118.0, -119.0]
})

# Make predictions on the new data
new_pred = lr.predict(new_data)
print("Predictions on New Data:")
print(new_pred)