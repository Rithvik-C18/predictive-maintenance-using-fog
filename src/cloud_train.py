import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import MODEL_DIR, MODEL_FILE
from data_utils import load_dataset, split_features_labels, train_test_split_data


def train_model(save_model=True, report=True):
    df = load_dataset()
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if report:
        report_text = classification_report(y_test, y_pred, digits=4)
        print(report_text)

    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_FILE)
        print(f"Saved model to {MODEL_FILE}")

    return pipeline


if __name__ == "__main__":
    train_model()
