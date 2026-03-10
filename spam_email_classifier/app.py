import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import numpy as np
from src.inference import predict_spam

st.set_page_config(page_title="Spam Email Classifier", layout="wide")
st.title("🔍 Spam Email Classifier (UCI Spambase)")

st.markdown("Move the sliders to simulate different feature values and see spam/ham prediction.")

col1, col2, col3 = st.columns(3)
with col1:
    free = st.slider("word_freq_free", 0.0, 5.0, 0.1, 0.1)
with col2:
    your = st.slider("word_freq_your", 0.0, 5.0, 0.1, 0.1)
with col3:
    cap_run = st.slider("capital_run_length", 0.0, 100.0, 5.0, 1.0)

features = np.zeros(57, dtype=float)
features[0] = free
features[4] = your
features[56] = cap_run

if st.button("🔍 Predict", type="primary"):
    label, spam_p, ham_p = predict_spam(features.tolist())
    st.subheader(label)
    c1, c2 = st.columns(2)
    c1.metric("Spam probability", f"{spam_p:.1%}")
    c2.metric("Ham probability", f"{ham_p:.1%}")

st.caption("Model: Numpy Naive Bayes trained on UCI Spambase dataset.")
