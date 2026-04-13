# Spam Email Classifier

A Machine Learning-based web application that classifies messages as Spam or Not Spam using Natural Language Processing techniques.

---

## Overview

This project uses NLP preprocessing and a Naive Bayes model to classify messages. It also includes a Streamlit web interface for real-time predictions.

---

## Features

- Text preprocessing using NLTK
- TF-IDF vectorization
- Naive Bayes classification
- Real-time prediction via web interface
- Confidence score display

---

## Model Performance

- Accuracy: ~98%
- High precision for spam detection

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Streamlit

---

## Project Structure

```
Spam Email Checker/
│
├── data/
├── notebook/
├── app.py
├── model.pkl
├── tfidf.pkl
├── requirements.txt
└── README.md
```
---

---

## How to Run Locally

1. Clone the repository

git clone https://github.com/yourusername/spam-classifier.git

2. Install Dependencies
   pip install -r requirements.txt

3. Run the App
   streamlit run app.py

---

## Example

Input:
Free entry in a weekly competition

Output:
Spam

---

## Author

Arnav Priyadarshi
