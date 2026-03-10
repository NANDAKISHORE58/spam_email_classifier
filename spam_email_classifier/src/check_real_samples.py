import pandas as pd
from inference import predict_spam

# Load processed test data
test = pd.read_csv('data/processed/test.csv', header=None).values

# One real ham sample (label = 0)
ham_sample = test[test[:, -1] == 0][0]
ham_features = ham_sample[:-1].tolist()

# One real spam sample (label = 1)
spam_sample = test[test[:, -1] == 1][0]
spam_features = spam_sample[:-1].tolist()

print("=== REAL HAM ROW (label 0) ===")
label, spam_p, ham_p = predict_spam(ham_features)
print("Pred:", label, f"(Spam: {spam_p:.2%}, Ham: {ham_p:.2%})")

print("\n=== REAL SPAM ROW (label 1) ===")
label, spam_p, ham_p = predict_spam(spam_features)
print("Pred:", label, f"(Spam: {spam_p:.2%}, Ham: {ham_p:.2%})")
