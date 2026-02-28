import joblib
import pandas as pd

from config import MODEL_FILE
from data_utils import align_features, load_dataset, split_features_labels


def edge_predict(sample):
    payload = joblib.load(MODEL_FILE)
    pipeline = payload["pipeline"]
    feature_names = payload["feature_names"]
    sample_aligned = align_features(sample, feature_names)
    pred = pipeline.predict(sample_aligned)
    return int(pred[0])


def edge_predict_stream(row, feature_names, pipeline):
    sample = pd.DataFrame([row])
    sample_aligned = align_features(sample, feature_names)
    pred = pipeline.predict(sample_aligned)
    return int(pred[0])


def main():
    df = load_dataset()
    X, _ = split_features_labels(df)

    sample = X.iloc[[0]]
    label = edge_predict(sample)

    status = "Faulty" if label == 1 else "Healthy"
    print(f"Edge prediction: {label} ({status})")


if __name__ == "__main__":
    main()
