from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_utils import load_dataset, split_features_labels, train_test_split_data
from cloud_train import train_model


def evaluate():
    df = load_dataset()
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Train a fresh model for evaluation
    pipeline = train_model(save_model=False, report=False)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate()
