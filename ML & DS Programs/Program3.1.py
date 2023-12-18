import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

iris = load_iris()
x=iris.data
y=iris.target

iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df['target'] = iris.target
print(iris_df)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42)

decision_tree = DecisionTreeClassifier(max_depth=3)

decision_tree.fit(x_train,y_train)
predict=decision_tree.predict(x_test)

result=accuracy_score(y_test,predict)
report=classification_report(y_test,predict)

print("\nAccuracy Score : ",result)
print("\nClassification Report : \n",report)


plot_tree(decision_tree, feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.title("Decision Tree Algorithm")
plt.show()

new_data = [[5.5,6.6,2.4,3.1],[1.3,4.5,5.6,7.0]]

new_data_predict = decision_tree.predict(new_data)

print("\nPrediction with new data\n",new_data_predict)





































