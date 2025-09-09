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
    # لو الموديل والفيكتورايزر موجودين بالفعل
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer

    # لو مش موجودين → هندرب من news.csv
    if not os.path.exists(DATA_FILE):
        st.error("⚠️ ملف البيانات news.csv مش موجود في مجلد data/.")
        st.stop()

    st.info("⏳ جاري تدريب الموديل لأول مرة...")

    # تحميل البيانات
    df = pd.read_csv(DATA_FILE)

    # تأكد إن الأعمدة مظبوطة
    if "text" not in df.columns or "label" not in df.columns:
        st.error("⚠️ لازم يكون في أعمدة: text و label في news.csv")
        st.stop()

    X = df["text"]
    y = df["label"]

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # تحويل النصوص
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # تدريب الموديل
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    # دقة مبدئية
    acc = accuracy_score(y_test, model.predict(X_test_vec))
    st.success(f"✅ تدريب ناجح! دقة مبدئية: {acc:.2f}")

    # حفظ الملفات
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return model, vectorizer


# ================================
# واجهة Streamlit
# ================================
st.title("📰 Fake News Detection")
st.write("تحقق من مصداقية الأخبار باستخدام نموذج تعلم الآلة.")

# تحميل أو تدريب الموديل
model, vectorizer = load_or_train_model()

# إدخال المستخدم
user_input = st.text_area("✍️ اكتب الخبر هنا للتحقق:")

if st.button("تحقق"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.error("🚨 هذا الخبر يبدو **كاذبًا**")
        else:
            st.success("✅ هذا الخبر يبدو **حقيقيًا**")
    else:
        st.warning("⚠️ من فضلك أدخل نص الخبر أولاً.")
