import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory as this script.")
    st.stop()

st.title("Email Spam Classifier")
st.write("The app is running. Please enter a message below.")

input = st.text_area("Enter the message")
if st.button("Predict"):
    if input:
        transformed_email = transform_text(input)
        vector_input = tfidf.transform([transformed_email])
        try:
            result = model.predict(vector_input)[0]
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not spam")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Please enter a message to classify.")
