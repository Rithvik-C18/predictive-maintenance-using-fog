# Industry IoT Predictive Maintenance (Edge/Fog/Cloud)

This project implements a lightweight predictive maintenance pipeline inspired by the "Fog and Edge Computing - DA 1" specification. It simulates a layered architecture:

- **Edge**: fast, low-power inference with logistic regression on sensor readings
- **Fog**: aggregation and system-wide health analytics
- **Cloud**: periodic retraining and model persistence

## Project Structure

```
.
|-- data/                 # dataset and generated artifacts
|-- src/
|   |-- edge_inference.py  # edge layer inference logic
|   |-- fog_aggregator.py  # fog aggregation logic
|   |-- cloud_train.py     # cloud model training pipeline
|   |-- evaluate.py        # evaluation metrics
|   |-- data_utils.py      # dataset utilities
|   `-- config.py          # configuration
`-- requirements.txt
```

## Quick Start

1. Install dependencies

```
pip install -r requirements.txt
```

2. Place the Kaggle dataset CSV in `data/` (name: `predictive_maintenance.csv`).

3. Train a model and save it:

```
python src/cloud_train.py
```

4. Run edge inference (simulated) on a sample record:

```
python src/edge_inference.py
```

5. Aggregate multiple edge outputs (simulated):

```
python src/fog_aggregator.py
```

6. Evaluate model performance:

```
python src/evaluate.py
```

## Notes

- The project uses a binary classifier: **0 = Healthy**, **1 = Faulty**.
- If you use a different CSV filename, update `src/config.py`.
