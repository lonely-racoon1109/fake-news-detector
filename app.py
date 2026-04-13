import streamlit as st
import pickle
import re

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Text preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Prediction function
def predict_news(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    result = model.predict(vec)[0]
    score = model.decision_function(vec)[0]
    return result, score

# Streamlit UI
st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detection System")
st.subheader("Enter a news article to check whether it is REAL or FAKE.")

user_input = st.text_area(" ", height=300)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result, score = predict_news(user_input)

        if result == "REAL":
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")

        st.write(f"Confidence score: {round(score, 2)}")