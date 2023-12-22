from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as np

iris = load_iris()
x=iris.data
y=iris.target

iris_df = np.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target']= iris.target
np.set_option('display.max_columns',None)
print(iris_df)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)

predict = knn.predict(x_test)
result = accuracy_score(y_test,predict)

print("Accuracy Score = ",result)

plt.scatter(x_test[:,0],x_test[:,1],c=predict,edgecolors='k')
plt.title="KNN ALGORITHM"
plt.show()