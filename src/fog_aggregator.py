from collections import Counter
import random


def simulate_edge_predictions(n=10):
    return [random.choice([0, 1]) for _ in range(n)]


def aggregate(preds):
    counts = Counter(preds)
    total = sum(counts.values())
    faulty_rate = counts.get(1, 0) / total if total else 0
    return {
        "total": total,
        "healthy": counts.get(0, 0),
        "faulty": counts.get(1, 0),
        "faulty_rate": faulty_rate,
    }


def main():
    preds = simulate_edge_predictions(n=20)
    summary = aggregate(preds)

    print("Fog aggregation summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
