from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
bc=load_breast_cancer()
x=bc.data
y=bc.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
gnb=GaussianNB()
gnb.fit(x_train,y_train)
print(gnb.predict(x_test))
G=gnb.predict(x_test)
result=accuracy_score(y_test,G)
print("Accuracy= ",result)
cr=classification_report(y_test,G)
print("/n Classification Report: ",cr)
