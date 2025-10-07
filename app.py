import streamlit as st 
import pickle
import sklearn
from nltk.corpus import stopwords
import nltk 
import os
import string 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
# Make sure nltk looks inside your project folder for data
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download required resources to that folder
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)



def transform_text(text):
    text = text.lower()
    
    text= nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and  i not in string.punctuation:
            y.append(i)
            
    text = y[:]       
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)



tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message ')

option = st.selectbox("You Got Message From :-", ["Via Email ", "Via SMS", "other"])


if st.checkbox("Check me"):
    st.write("")

if st.button('Click to Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]


    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')



