
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# reading the data
data = pd.read_csv("SMSSpamCollection", delimiter='\t')
data.columns = ['class','text']
X = data.text

y = data['class'].apply(lambda i: 1 if i == 'ham' else 0)
#we split the data before vectorization
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

vectorizerC = CountVectorizer()
X_train_document_term_matrix = vectorizerC.fit_transform(X_train)
X_test_document_term_matrix = vectorizerC.transform(X_test)

# The common method for text classification is naive bayes
clf = MultinomialNB()
clf.fit(X_train_document_term_matrix,y_train)
y_prediction_class = clf.predict(X_test_document_term_matrix)

print("Prediction Accuracy : ", accuracy_score(y_test,y_prediction_class ))

conf_mat = metrics.confusion_matrix(y_test,y_prediction_class)
print("Confusion matrix :" , conf_mat)
