import os
os.system("pip install nltk")

import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords

# Ensure stopwords are available (safe for deployment)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load model & vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Page config
st.set_page_config(
    page_title="Spam Email Classifier",
    layout="centered"
)

# Header
st.markdown("<h1 style='text-align: center;'>Spam Email Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect whether a message is Spam or Not</p>", unsafe_allow_html=True)
st.markdown("---")

# Input
message = st.text_area(
    "Enter your message here:",
    height=150,
    placeholder="Type your message..."
)

# Buttons
col1, col2 = st.columns(2)

with col1:
    check = st.button("Analyze")

with col2:
    clear = st.button("Clear")

# Clear functionality
if clear:
    st.session_state.clear()
    st.rerun()

# Prediction
if check:
    if message:
        processed = preprocess_text(message)
        vector = tfidf.transform([processed]).toarray()
        result = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0][1]

        st.markdown("---")

        if result == 1:
            st.error("Result: Spam")
            st.write(f"Confidence Score: {prob:.2f}")
        else:
            st.success("Result: Not Spam")
            st.write(f"Confidence Score: {1 - prob:.2f}")
    else:
        st.warning("Please enter a message")

# Examples section
st.markdown("---")
st.subheader("Try Examples")

col3, col4 = st.columns(2)

with col3:
    if st.button("Spam Example"):
        st.info("Free entry in a weekly competition win now")

with col4:
    if st.button("Normal Example"):
        st.info("Hey, are we meeting today?")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built by Arnav Priyadarshi</p>",
    unsafe_allow_html=True
)