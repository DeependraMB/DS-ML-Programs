from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report


categories = ['soc.religion.christian','sci.med','alt.atheism','sci.space']
newsgroups = fetch_20newsgroups(subset='train',categories=categories,shuffle=True)

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

svm_classifier = SVC(kernel='linear',C=1.0)

svm_classifier.fit(x_train,y_train)

predictions = svm_classifier.predict(x_test)

acc = accuracy_score(y_test,predictions)
rep = classification_report(y_test,predictions)

print("ACCURACY:",acc)
print("Classification Report:",rep)

new_data = ["christian"]

new_data_Tfid = vectorizer.transform(new_data)

new_prediction = svm_classifier.predict(new_data_Tfid)

print(new_prediction)

for i, prediction in enumerate(new_prediction):
    prediction_category = newsgroups.target_names[new_prediction[i]]
    print(prediction_category)




