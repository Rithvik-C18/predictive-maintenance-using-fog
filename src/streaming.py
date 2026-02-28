import argparse
import time

import joblib

from config import MODEL_FILE
from data_utils import load_dataset, split_features_labels
from edge_inference import edge_predict_stream
from fog_aggregator import aggregate


def stream_inference(interval_sec=0.2, window_size=20, max_rows=None):
    payload = joblib.load(MODEL_FILE)
    pipeline = payload["pipeline"]
    feature_names = payload["feature_names"]

    df = load_dataset()
    X, _ = split_features_labels(df)

    buffer = []
    count = 0
    for _, row in X.iterrows():
        pred = edge_predict_stream(row, feature_names, pipeline)
        buffer.append(pred)
        print(f"edge_pred: {pred}")
        count += 1

        if len(buffer) >= window_size:
            summary = aggregate(buffer)
            print("fog_window_summary:", summary)
            buffer = []

        if max_rows is not None and count >= max_rows:
            break

        time.sleep(interval_sec)

    if buffer:
        summary = aggregate(buffer)
        print("fog_window_summary:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    stream_inference(
        interval_sec=args.interval,
        window_size=args.window_size,
        max_rows=args.max_rows,
    )
