import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# ---------- Paths ----------
# This file is: D:\spam_email_classifier\src\text_model_training.py
# Project root: D:\spam_email_classifier
SRC_DIR = os.path.dirname(os.path.abspath(__file__))              # ...\spam_email_classifier\src
BASE_DIR = os.path.dirname(SRC_DIR)                               # ...\spam_email_classifier
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "sms_spam.csv") # ...\data\raw\sms_spam.csv
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("DATA_PATH =", DATA_PATH)
print("Exists?", os.path.exists(DATA_PATH))

# ---------- Load UCI-format file (tab-separated) ----------
df = pd.read_csv(
    DATA_PATH,
    sep="\t",          # UCI format: label<TAB>text[web:17][web:99]
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)

df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["ham", "spam"])].copy()
df = df[df["text"].notna()]
df["text"] = df["text"].astype(str)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

print("Data used for training (head):")
print(df.head())
print("Total rows:", len(df))

# ---------- Train ----------
X = df["text"]
y = df["label_num"]

vectorizer = CountVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

clf = MultinomialNB()
clf.fit(X_vec, y)

y_pred = clf.predict(X_vec)
acc = accuracy_score(y, y_pred)
print(f"Training accuracy on {len(df)} samples: {acc:.3f}")

# ---------- Save model ----------
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "text_vectorizer.pkl"))
joblib.dump(clf, os.path.join(MODELS_DIR, "text_nb.pkl"))
print("✅ Saved text_vectorizer.pkl and text_nb.pkl in models/")
