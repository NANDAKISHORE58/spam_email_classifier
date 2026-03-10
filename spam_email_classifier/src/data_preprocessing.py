import pandas as pd
import numpy as np
import os

DATA_RAW = 'data/raw/spambase_clean.data'
DATA_PROCESSED = 'data/processed'

df = pd.read_csv(DATA_RAW, header=None)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

os.makedirs(DATA_PROCESSED, exist_ok=True)

np.random.seed(42)
indices = np.arange(len(df))
np.random.shuffle(indices)
split_idx = int(0.8 * len(indices))

train_data = df.iloc[indices[:split_idx]]
test_data = df.iloc[indices[split_idx:]]

train_data.to_csv(os.path.join(DATA_PROCESSED, 'train.csv'), index=False, header=False)
test_data.to_csv(os.path.join(DATA_PROCESSED, 'test.csv'), index=False, header=False)

print(f"Processed: {len(train_data)} train, {len(test_data)} test")
