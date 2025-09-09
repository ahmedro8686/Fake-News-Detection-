import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATA_FILE = "data/news.csv"


@st.cache_resource
def load_or_train_model():
    # If model and vectorizer already exist
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer

    # If they don't exist → train from news.csv
    if not os.path.exists(DATA_FILE):
        st.error("⚠️ news.csv file not found in data/ directory.")
        st.stop()

    st.info("⏳ Training the model for the first time...")

    # Load dataset
    df = pd.read_csv(DATA_FILE)

    # Ensure required columns exist
    if "text" not in df.columns or "label" not in df.columns:
        st.error("⚠️ The dataset must contain 'text' and 'label' columns.")
        st.stop()

    X = df["text"]
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert text to TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    # Evaluate accuracy
    acc = accuracy_score(y_test, model.predict(X_test_vec))
    st.success(f"✅ Training completed! Initial Accuracy: {acc:.2f}")

    # Save model and vectorizer
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return model, vectorizer


# ================================
# Streamlit UI
# ================================
st.title("📰 Fake News Detection")
st.write("Check if a news article is real or fake using Machine Learning.")

# Load or train model
model, vectorizer = load_or_train_model()

# User input
user_input = st.text_area("✍️ Enter the news text to verify:")

if st.button("Verify"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.error("🚨 This news seems **Fake**")
        else:
            st.success("✅ This news seems **Real**")
    else:
        st.warning("⚠️ Please enter some text first.")
