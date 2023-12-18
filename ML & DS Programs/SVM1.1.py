from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC

newsgroups = fetch_20newsgroups(subset='all',remove=('headers','footers','quotes'))

for filename in newsgroups.filenames:
    print(filename)

categories = ["sci.electronics","sci.med","sci.crypt","rec.motorcycles"]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

svm_classifier = SVC(kernel='linear',C=1.0)

svm_classifier.fit(x_train,y_train)
prediction = svm_classifier.predict(x_test)

result = accuracy_score(y_test,prediction)
report = classification_report(y_test,prediction)

print("Accuracy Score  =  ", result)
print("Classification report = \n",report)

new_data = ["electronics"]

x_newdata_tfid = vectorizer.transform(new_data)

new_prediction = svm_classifier.predict(x_newdata_tfid)

for i , prediction in enumerate(new_data):
    predicted_category = newsgroups.target_names[new_prediction[i]]
    print(predicted_category)