from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sb
import pandas as pd

bc = load_breast_cancer()
x = bc.data
y = bc.target

df = pd.DataFrame(data=bc.data,columns=bc.feature_names)
df['target'] = bc.target
print(df)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
naive = GaussianNB()
naive.fit(x_train,y_train)

predict = naive.predict(x_test)
result = accuracy_score(y_test,predict)
report= classification_report(y_test,predict)
conf =confusion_matrix(y_test,predict)

print("Accuracy Score = ",result)
print("Classification Report= ",report)
print("Confusion Matrix :\n",conf)

sb.heatmap(conf,annot=True,cmap="Blues", fmt='g')
plt.show()