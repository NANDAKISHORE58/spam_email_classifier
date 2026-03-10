import joblib
import numpy as np
import os
from src.model import NumpyNB  # <-- IMPORTANT

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def prepare_sample(features_list):
    sample = np.array(features_list[:57], dtype=np.float32)
    if len(sample) < 57:
        sample = np.pad(sample, (0, 57 - len(sample)), 'constant')
    return sample.reshape(1, -1)

def predict_spam(features_list):
    model_path = os.path.join(MODELS_DIR, 'spam_nb.pkl')
    model = joblib.load(model_path)
    sample = prepare_sample(features_list)
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]
    label = 'Spam 🛑' if pred == 1 else 'Ham ✅'
    return label, prob[1], prob[0]

if __name__ == "__main__":
    features = [0.1, 0.2, 0.0] * 19
    label, spam_p, ham_p = predict_spam(features)
    print(f"{label} (Spam: {spam_p:.2%}, Ham: {ham_p:.2%})")
