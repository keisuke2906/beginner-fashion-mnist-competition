#!/usr/bin/env python3
"""Load trained model pickle and report validation results."""

import csv
import pickle
from pathlib import Path

import numpy as np

from load_fashion_mnist import load_eval_data
from network import SimpleMLP

WEIGHTS_PATH = Path("sample_weight.pkl")
PREDICTION_CSV = Path("validation_predictions.csv")


def main() -> int:
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] weights file not found: {WEIGHTS_PATH}")
        return 1

    with WEIGHTS_PATH.open("rb") as f:
        state = pickle.load(f)

    model = SimpleMLP.from_state(state)

    x_valid, t_valid = load_eval_data()

    pred = model.predict(x_valid)
    acc = float(np.mean(pred == t_valid))
    print(f"Validation Accuracy: {acc:.6f}")

    with PREDICTION_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "prediction", "correct"])
        for idx, (label, prediction) in enumerate(zip(t_valid, pred)):
            writer.writerow([idx, int(label), int(prediction), int(label == prediction)])

    print(f"Saved validation predictions: {PREDICTION_CSV.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
