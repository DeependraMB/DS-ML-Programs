from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score ,classification_report
categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)
vectorizer=TfidfVectorizer()
X_train_tfidf=vectorizer.fit_transform(twenty_train.data)
y_train = twenty_train.target

x_train,x_test,y_train,y_test=train_test_split(X_train_tfidf,y_train,test_size=0.2,random_state=42)

svm_classifier=SVC(kernel='linear',random_state=42)
svm_classifier.fit(x_train,y_train)
predictions=svm_classifier.predict(x_test)
accuracy=accuracy_score(y_test,predictions)
classification=classification_report(y_test,predictions,target_names=twenty_train.target_names)

print("Accuracy",accuracy)
print("classification_report:",classification)
new_data=["I have a question about computer graphics","This is a medical related topic"]
x_new_tfidf=vectorizer.transform(new_data)
new_predictions=svm_classifier.predict(x_new_tfidf)

for i,text in enumerate(new_data):
 predicted_category=twenty_train.target_names[new_predictions[i]]
 print("Predicted Cate",predicted_category)

