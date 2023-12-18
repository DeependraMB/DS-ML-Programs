from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

bc=load_breast_cancer()

x=bc.data
y=bc.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
knn=KNeighborsClassifier(n_neighbors=17)

knn.fit(x_train,y_train)
result=knn.predict(x_test)
acc=accuracy_score(y_test,result)

print(result,acc)
