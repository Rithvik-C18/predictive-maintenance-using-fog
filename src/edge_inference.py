import joblib
from config import MODEL_FILE
from data_utils import load_dataset, split_features_labels


def edge_predict(sample):
    model = joblib.load(MODEL_FILE)
    pred = model.predict(sample)
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
