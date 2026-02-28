import json

from pathlib import Path

import joblib

from config import MODEL_FILE


def export_tinyml(out_path):
    payload = joblib.load(MODEL_FILE)
    pipeline = payload["pipeline"]

    scaler = pipeline.named_steps.get("scaler")
    model = pipeline.named_steps.get("model")

    if scaler is None or model is None:
        raise ValueError("Pipeline must include scaler and model steps")

    export = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print(f"Exported TinyML parameters to {out_path}")


if __name__ == "__main__":
    export_tinyml("data/tinyml_params.json")
