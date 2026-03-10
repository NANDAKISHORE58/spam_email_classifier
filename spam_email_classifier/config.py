import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'spambase.data')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# Model config
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = 57  # Last column is spam label (0=ham, 1=spam)

# UCI Spambase has 58 columns: 57 features + 1 label
COLUMN_NAMES = [f'feature_{i}' for i in range(58)]
