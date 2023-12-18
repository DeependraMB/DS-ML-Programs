from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
naive = GaussianNB()
naive.fit(x_train,y_train)

predict = naive.predict(x_test)
result = accuracy_score(y_test,predict)
report= classification_report(y_test,predict)

print("Accuracy Score = ",result)
print("Classification Report= ",report)

new_data = [[5.0, 4.5, 5.6, 7.0],[3.0,2.0,4.5,5.5]]

y_pred_new_data = naive.predict(new_data)

for i, prediction in enumerate(y_pred_new_data):
    print(f"Data point {i + 1}: Predicted Class {prediction}")


# Visualize the test set predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_test[:, 0], x_test[:, 1], c=predict, cmap=plt.cm.Paired, edgecolor='k')
plt.title('Test Set Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')



plt.show()