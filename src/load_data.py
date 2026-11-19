"""Load the ML-ready data: time-based splits with the saved preprocessing pipeline applied."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_dataset.parquet"
PIPELINE = ROOT / "models" / "preprocessor.joblib"
SUMMARY = ROOT / "data" / "metadata" / "preprocessing_summary.json"


def load_splits(include_held_out: bool = False):
    """Return ({name: (X, y, cve_ids)}, fitted_preprocessor). X is a sparse matrix, y is 0/1 (is_kev)."""
    info = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cols = info["numeric_features"] + info["categorical_features"]
    df = pd.read_parquet(DATA)
    for col in info["categorical_features"]:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    pre = joblib.load(PIPELINE)
    split, year = info["split"], df["published_year"]
    masks = {
        "train": year.between(*split["train_years"]),
        "validation": year == split["validation_year"],
        "test": year == split["test_year"],
    }
    if include_held_out:
        masks["held_out"] = year > split["test_year"]

    out = {}
    for name, mask in masks.items():
        part = df[mask]
        out[name] = (pre.transform(part[cols]), part["is_kev"].to_numpy(), part["cve_id"].to_numpy())
    return out, pre


if __name__ == "__main__":
    splits, pre = load_splits(include_held_out=True)
    for name, (X, y, ids) in splits.items():
        print(f"{name:<11} X {X.shape} | rows {len(y):,} | KEV {int(y.sum()):,} ({y.mean() * 100:.3f}%)")