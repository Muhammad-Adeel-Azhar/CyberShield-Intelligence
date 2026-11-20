"""Day 3, Part A, Model 2 (step 1): tune the Decision Tree on the validation year (2023). The 2024 test set is NOT used."""
import json
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

from src.load_data import load_splits
from src.train_knn import evaluate   # same metrics as for kNN, so the two models are compared fairly

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "metadata"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

DEPTHS = [2, 3, 4, 6, 8, 12, 16, None]          # None = grow until the leaves are pure or too small
LEAVES = [1, 5, 20, 50, 100, 200]               # minimum samples per leaf
CLASS_WEIGHTS = [None, "balanced"]
SEED = 42


def fit_one(depth, leaf, cw, X_train, y_train, X_val, y_val) -> dict:
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight=cw, random_state=SEED)
    tree.fit(X_train, y_train)
    p_val = tree.predict_proba(X_val)[:, 1]
    p_train = tree.predict_proba(X_train)[:, 1]
    return {
        "max_depth": "None" if depth is None else str(depth),
        "min_samples_leaf": leaf,
        "class_weight": "none" if cw is None else cw,
        "depth_reached": int(tree.get_depth()),
        "leaves": int(tree.get_n_leaves()),
        "train_avg_precision": round(float(average_precision_score(y_train, p_train)), 4),
        **evaluate(y_val, p_val),
    }


def main() -> None:
    splits, _ = load_splits()
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["validation"]
    X_train, X_val = X_train.toarray().astype(np.float32), X_val.toarray().astype(np.float32)
    pd.set_option("display.width", 220, "display.max_columns", None, "display.max_rows", None)

    print("1. Data (training years 2016-2022, validation year 2023; the test set is not loaded into any step)")
    print(f"train {X_train.shape} | KEV {int(y_train.sum()):,} ({y_train.mean() * 100:.3f}%)")
    print(f"val   {X_val.shape} | KEV {int(y_val.sum()):,} ({y_val.mean() * 100:.3f}%)")
    print(f"Baselines on validation: always 'not KEV' accuracy {(y_val == 0).mean():.5f} (recall 0) | "
          f"no-skill average precision {y_val.mean():.4f}")

    combos = list(product(DEPTHS, LEAVES, CLASS_WEIGHTS))
    print(f"\n2. Fitting {len(combos)} trees (max_depth x min_samples_leaf x class_weight) on all CPU cores")
    t0 = time.time()
    rows = Parallel(n_jobs=-1)(delayed(fit_one)(d, l, c, X_train, y_train, X_val, y_val) for d, l, c in combos)
    print(f"done in {time.time() - t0:.0f} s")

    grid = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    grid.to_csv(REPORTS / "tree_validation_grid.csv", index=False)
    depth_order = [str(d) for d in DEPTHS]

    print("\n3. Validation average precision (higher is better) by max_depth (rows) and min_samples_leaf (columns)")
    for cw in ("none", "balanced"):
        part = grid[grid["class_weight"] == cw]
        table = part.pivot(index="max_depth", columns="min_samples_leaf", values="avg_precision").reindex(depth_order)
        print(f"\nclass_weight = {cw}")
        print(table.round(4).to_string())

    print("\n4. Overfitting check: training vs validation average precision (min_samples_leaf=1)")
    over = grid[grid["min_samples_leaf"] == 1][["class_weight", "max_depth", "leaves", "train_avg_precision", "avg_precision"]]
    over = over.rename(columns={"avg_precision": "val_avg_precision"})
    print(over.to_string(index=False))

    print("\n5. Top 10 configurations by validation average precision")
    cols = ["max_depth", "min_samples_leaf", "class_weight", "leaves", "avg_precision", "roc_auc", "accuracy",
            "precision_at_0.5", "recall_at_0.5", "f1_at_0.5", "best_f1", "best_f1_threshold"]
    top = grid.sort_values("avg_precision", ascending=False)
    print(top[cols].head(10).to_string(index=False))

    print("\n6. Effect of class weighting: best configuration of each kind")
    for cw in ("none", "balanced"):
        b = grid[grid["class_weight"] == cw].sort_values("avg_precision", ascending=False).iloc[0]
        print(f"{cw:<9}: depth {b['max_depth']}, leaf {b['min_samples_leaf']} | AP {b['avg_precision']:.4f} | "
              f"at 0.5: precision {b['precision_at_0.5']:.4f}, recall {b['recall_at_0.5']:.4f}, F1 {b['f1_at_0.5']:.4f} | "
              f"best F1 {b['best_f1']:.4f}")

    best = top.iloc[0]
    print("\n7. Selected configuration (highest validation average precision, same rule as kNN)")
    print(f"max_depth={best['max_depth']} | min_samples_leaf={best['min_samples_leaf']} | class_weight={best['class_weight']}")
    print(f"average precision {best['avg_precision']:.4f} | ROC AUC {best['roc_auc']:.4f} | leaves {best['leaves']}")
    print(f"Plain 0.5 rule: precision {best['precision_at_0.5']:.4f}, recall {best['recall_at_0.5']:.4f}, "
          f"F1 {best['f1_at_0.5']:.4f}")
    print(f"Best-F1 threshold {best['best_f1_threshold']}: precision {best['best_f1_precision']:.4f}, "
          f"recall {best['best_f1_recall']:.4f}, F1 {best['best_f1']:.4f}")

    # Figure: AP against depth, one line per leaf size, one panel per class weighting
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, cw in zip(axes, ("none", "balanced")):
        part = grid[grid["class_weight"] == cw]
        for leaf in LEAVES:
            line = part[part["min_samples_leaf"] == leaf].set_index("max_depth").reindex(depth_order)
            ax.plot(depth_order, line["avg_precision"], marker="o", label=f"leaf >= {leaf}")
        ax.axhline(y_val.mean(), color="grey", linestyle="--", label="no-skill")
        ax.set(title=f"Decision Tree on validation (2023), class_weight = {cw}", xlabel="max_depth",
               ylabel="Average precision")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "day3_tree_validation.png", dpi=150)
    plt.close(fig)

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "grid": {"max_depth": [str(d) for d in DEPTHS], "min_samples_leaf": LEAVES,
                 "class_weight": ["none", "balanced"]},
        "random_state": SEED,
        "selection_rule": "highest validation average precision",
        "selected": {"max_depth": str(best["max_depth"]), "min_samples_leaf": int(best["min_samples_leaf"]),
                     "class_weight": str(best["class_weight"]),
                     "threshold_best_f1_on_validation": float(best["best_f1_threshold"])},
        "validation_metrics_selected": {k: (v if isinstance(v, str) else float(v)) for k, v in best.items()},
        "test_set_used": False,
    }
    (META_DIR / "tree_tuning.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved reports/tree_validation_grid.csv, {FIGURES / 'day3_tree_validation.png'} and data/metadata/tree_tuning.json")


if __name__ == "__main__":
    main()
