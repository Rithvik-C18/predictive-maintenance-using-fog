import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_FILE, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE


def load_dataset(path=DATA_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def split_features_labels(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def train_test_split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )
