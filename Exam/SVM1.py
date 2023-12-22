from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report

categories = ['comp.graphics','sci.med','talk.politics.mideast','sci.electronics']
newsgroups = fetch_20newsgroups(subset="train",categories=categories,shuffle=True,random_state=42)

# for i in newsgroups.filenames:
#     print(i)



vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

svm_classifier = SVC(kernel='linear',C=1.0)

svm_classifier.fit(x_train,y_train)

prediction = svm_classifier.predict(x_test)
result = accuracy_score(y_test,prediction)
report = classification_report(y_test,prediction)

print("Accuracy Score :",result)
print("Classification Report :",report)

new_data = ["Politics"]

new_data_Tfid = vectorizer.transform(new_data)
new_prediction = svm_classifier.predict(new_data_Tfid)

for i,prediction in enumerate(new_data):
    new_category = newsgroups.target_names[new_prediction[i]]
    print(new_category)


