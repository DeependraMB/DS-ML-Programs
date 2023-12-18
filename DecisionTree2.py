
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris;
from sklearn.metrics import accuracy_score,classification_report;
from sklearn.model_selection import train_test_split;
import matplotlib.pyplot as mplb
from sklearn.tree import plot_tree

iris=load_iris();

x=iris.data;
y=iris.target;

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42);
naive=DecisionTreeClassifier(max_depth=8);

naive.fit(x_train,y_train);
v=naive.predict(x_test);

result=accuracy_score(y_test,v);
report=classification_report(y_test,v);


print(result,report);

mplb.figure(figsize=(20,30))
plot_tree(naive,class_names=iris.target_names,feature_names=iris.feature_names,rounded=True,filled=True)
mplb.show()


