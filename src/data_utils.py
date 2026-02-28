import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_FILE, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE


def load_dataset(path=DATA_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def select_features(df):
    if TARGET_COLUMN in df.columns:
        X = df.drop(columns=[TARGET_COLUMN])
    else:
        X = df.copy()

    X = X.select_dtypes(include=["number"]).copy()
    if X.empty:
        raise ValueError("No numeric feature columns found in dataset")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X


def split_features_labels(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    X = select_features(df)
    y = df[TARGET_COLUMN]
    return X, y


def align_features(df, feature_names):
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[feature_names].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X


def train_test_split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )
