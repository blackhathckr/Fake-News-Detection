import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import joblib as jb

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

jb.dump(model,'model.h5')

#print(pred)

# Confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


def classify(news):
    model=jb.load('model.h5')
    news = tfidf_vectorizer.transform([news]).toarray()
    classification = model.predict(news)
    st.title(str(classification[0]))

def fake_news_detection_webapp():

    st.title("Fake News Detection")
    news = st.text_area("Enter the News Article Headline : ")
    if st.button("Classify"):
        classify(news)
        
fake_news_detection_webapp()
        