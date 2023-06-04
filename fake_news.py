import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st

#Read the data
df=pd.read_csv('news.csv')

#Get shape and head
#df.shape
#df.head()

#Get the labels
labels=df.label
#labels.head()

# Split the dataset 
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2)

# Initialize a TfidfVectorizer Class Model
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the data
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier Class Model
model=PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
y_pred=model.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

input="Hillary Clinton is a liar"
input=tfidf_vectorizer.transform([input]).toarray()
pred=model.predict(input)
#print(pred)

# Confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

def fake_news_detection_webapp():
    st.title("Fake News Detection")

    input = st.text_area("Enter a News Article Headline: ")
    if len(input) < 1:
        st.write("  ")
    else:
        sample = input
        data = tfidf_vectorizer.transform([sample]).toarray()
        label = model.predict(data)
        st.title(str(label))

fake_news_detection_webapp()
        