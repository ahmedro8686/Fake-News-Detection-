import streamlit as st
import joblib

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")       # trained model
    vectorizer = joblib.load("vectorizer.pkl")  # TF-IDF vectorizer
    return model, vectorizer

model, vectorizer = load_model()

# Page setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detection App")
st.markdown(
    """
    This app uses **Machine Learning** to detect whether a news article is real or fake.  
    Enter your news text below and click **Check**.
    """
)

# User input
news_text = st.text_area("✍️ Enter news text:", height=200)

if st.button("Check ✅"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        # Transform text into TF-IDF features
        X_input = vectorizer.transform([news_text])
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        
        # Show result
        if prediction == 1:  # Real news
            st.success("✅ This news is classified as **REAL**.")
        else:  # Fake news
            st.error("❌ This news is classified as **FAKE**.")
        
        # Show confidence scores
        st.subheader("🔎 Confidence Score:")
        st.write(f"Real News: {probabilities[1]*100:.2f}%")
        st.write(f"Fake News: {probabilities[0]*100:.2f}%")

# Add example for testing
with st.expander("📌 Try an example"):
    st.write("The government announced a new healthcare reform plan today.")
    st.write("Scientists discovered that the moon is made of cheese.")
