import os
import sys

import streamlit as st
import joblib

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

MODELS_DIR = os.path.join(ROOT_DIR, "models")

st.set_page_config(page_title="Spam Email Classifier (Text)", layout="wide")
st.title("📧 Spam Email Classifier (Text-based)")

st.markdown("Type or paste an email/SMS message below and click **Classify**.")

user_text = st.text_area(
    "Email / SMS content",
    height=200,
    placeholder="e.g. Congratulations! You have won a $1000 Walmart gift card. Click here to claim now.",
)

if st.button("🔍 Classify"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        vectorizer_path = os.path.join(MODELS_DIR, "text_vectorizer.pkl")
        model_path = os.path.join(MODELS_DIR, "text_nb.pkl")

        if not (os.path.exists(vectorizer_path) and os.path.exists(model_path)):
            st.error("Model files not found. Run `python src/text_model_training.py` first.")
        else:
            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)

            X_vec = vectorizer.transform([user_text])
            prob = model.predict_proba(X_vec)[0]
            pred = model.predict(X_vec)[0]

            label = "Spam 🛑" if pred == 1 else "Ham ✅"

            st.subheader(label)
            c1, c2 = st.columns(2)
            c1.metric("Spam probability", f"{prob[1]:.1%}")
            c2.metric("Ham probability", f"{prob[0]:.1%}")
