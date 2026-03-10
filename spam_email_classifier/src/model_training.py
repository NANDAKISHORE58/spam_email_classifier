import numpy as np
import pandas as pd
import joblib
import os
from src.model import NumpyNB  # <-- IMPORTANT

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'train.csv'),
                    header=None).apply(pd.to_numeric, errors='coerce').fillna(0).values
test = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'test.csv'),
                   header=None).apply(pd.to_numeric, errors='coerce').fillna(0).values

X_train = train[:, :-1]
y_train = train[:, -1].round().astype(int)
X_test = test[:, :-1]
y_test = test[:, -1].round().astype(int)

nb = NumpyNB()
nb.fit(X_train, y_train)

preds = nb.predict(X_test)
acc = np.mean(preds == y_test)
print(f"Numpy NB Accuracy: {acc:.3f}")

joblib.dump(nb, os.path.join(MODELS_DIR, 'spam_nb.pkl'))
print("✅ Model saved to models/spam_nb.pkl")
