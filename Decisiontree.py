from sklearn.tree import DecisionTreeClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import accuracy_score,classification_report;
from sklearn.datasets import load_iris;


iris=load_iris()

x=iris.data;
y=iris.target;

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42);

dt = DecisionTreeClassifier(max_depth=2);
dt.fit(x_train,y_train);

v = dt.predict(x_test)
report = classification_report(y_test, v)
result = accuracy_score(y_test, v)

print("Accuracy:", result)
print("\nClassification Report:\n", report)

new_data = [[1.5, 1.0, 2.0, 1.0]]
predicted_value = dt.predict(new_data)
new_prediction_label = iris.target_names[predicted_value[0]]

print("Predicted Label:", new_prediction_label)
