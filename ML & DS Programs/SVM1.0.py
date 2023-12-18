from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

categories = ['alt.atherium', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

vectorizer = TfidfVectorizer()
x_train_tfid = vectorizer.fit_transform(newsgroups.data)
y_train = newsgroups.target

x_train, x_test, y_train, y_test = train_test_split(x_train_tfid, y_train, test_size=0.3, random_state=42)

svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(x_train, y_train)

prediction = svm_classifier.predict(x_test)
accuracy_score = accuracy_score(y_test, prediction)
classification_report = classification_report(y_test, prediction)

print("Accuracy score = ", accuracy_score)
print("Classification Report = ", classification_report)

new_data = [ "computer Graphics" ]

x_newdata_tfidf = vectorizer.transform(new_data)
new_prediction = svm_classifier.predict(x_newdata_tfidf)



for i,prediction in enumerate(new_data):
    predicted_category = newsgroups.target_names[new_prediction[i]]
    print(predicted_category)

new_data_df = pd.DataFrame({'Text': new_data, 'Predicted Category': new_prediction})
print("\nPredictions for New Data:\n", new_data_df)

# X = newsgroups.data
# y = newsgroups.target

# for filename in newsgroups.filenames:
#     print(filename)

# newsgroups_df = pd.DataFrame(data={'text': X, 'target': y})

# print(newsgroups_df.columns)

