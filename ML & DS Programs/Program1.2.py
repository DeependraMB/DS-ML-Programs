import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

knn = KNeighborsClassifier(n_neighbors=9)
naiveG = GaussianNB()
naiveC = CategoricalNB()

knn.fit(x_train,y_train)
predictionKNN = knn.predict(x_test)
print(predictionKNN)
result = accuracy_score(predictionKNN,y_test)
print("Accuracy score KNN :",result)

naiveG.fit(x_train,y_train)
predictionNaiveG = naiveG.predict(x_test)
print(predictionNaiveG)
resultNaiveG = accuracy_score(predictionNaiveG,y_test)
print("Accuracy score Gaussians",resultNaiveG)

naiveC.fit(x_train,y_train)
predictionNaiveC = naiveC.predict(x_test)
print(predictionNaiveC)
resultNaiveC = accuracy_score(predictionNaiveC,y_test)
print("Accuracy score Categorical:",resultNaiveC)

