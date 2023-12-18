from sklearn.naive_bayes import GaussianNB;
from sklearn.datasets import load_iris;
from sklearn.metrics import accuracy_score,classification_report;
from sklearn.model_selection import train_test_split;

iris=load_iris();

x=iris.data;
y=iris.target;

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42);
naive=GaussianNB();

naive.fit(x_train,y_train);
v=naive.predict(x_test);

result=accuracy_score(y_test,v);
report=classification_report(y_test,v);


print(result,report);