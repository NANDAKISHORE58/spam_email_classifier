import pandas as pd
import numpy as np
from config import COLUMN_NAMES

def extract_features(df):
    """Extract/pass-through features (UCI already vectorized)."""
    # Features are pre-extracted: word freqs, char freqs, capital runs
    features = df.iloc[:, :-1].values  # All but last column
    return features.astype(np.float32)

def prepare_sample(features_list):
    """For inference: pad/align to 57 features."""
    sample = np.array(features_list[:57], dtype=np.float32)
    if len(sample) < 57:
        sample = np.pad(sample, (0, 57 - len(sample)))
    return sample.reshape(1, -1)
