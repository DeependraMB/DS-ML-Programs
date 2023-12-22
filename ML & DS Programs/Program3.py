from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import tree

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the classifier
decision_tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = decision_tree.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# New data for prediction
new_data = [[5.0, 4.0, 1.5, 0.2], [6.5, 3.0, 5.5, 1.8]]

# Make predictions on the new data
y_pred_new_data = decision_tree.predict(new_data)

# Display predictions for the new data
print("\nPredictions for New Data:")
for i, prediction in enumerate(y_pred_new_data):
    print(f"Predicted Class {prediction}")

# Visualize the decision tree
plt.figure(figsize=(8, 6))

tree.plot_tree(decision_tree,class_names=iris.target_names,feature_names=iris.feature_names,filled=True)
plt.title("Decision Tree Visualization")
plt.show()
