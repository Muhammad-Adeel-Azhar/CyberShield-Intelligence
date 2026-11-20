"""Day 3, Part A, Model 1 (step 1): tune kNN on the validation year (2023). The 2024 test set is NOT used here."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from src.load_data import load_splits

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "metadata"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

K_VALUES = [1, 3, 5, 11, 21, 51, 101, 201, 301]   # odd values avoid tied votes
WEIGHTS = ["uniform", "distance"]


def vote_probability(neigh_y: np.ndarray, neigh_d: np.ndarray, k: int, weights: str) -> np.ndarray:
    """Share of (weighted) votes for KEV among the k nearest neighbours; same rule as scikit-learn."""
    y = neigh_y[:, :k]
    if weights == "uniform":
        return y.mean(axis=1)
    d = neigh_d[:, :k]
    zero = d == 0
    w = 1.0 / np.where(zero, 1.0, d)
    has_zero = zero.any(axis=1)
    w[has_zero] = zero[has_zero].astype(float)   # exact duplicates of the query decide alone
    return (w * y).sum(axis=1) / w.sum(axis=1)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    """Metrics at the default 0.5 vote, plus the best F1 over all thresholds (threshold-free AP and AUC)."""
    pred = p > 0.5
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    pr, rc, thr = precision_recall_curve(y, p)
    f1_curve = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    best = int(f1_curve.argmax())
    return {
        "accuracy": round(float((pred == (y == 1)).mean()), 5),
        "precision_at_0.5": round(prec, 4),
        "recall_at_0.5": round(rec, 4),
        "f1_at_0.5": round(f1, 4),
        "avg_precision": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "best_f1": round(float(f1_curve[best]), 4),
        "best_f1_threshold": round(float(thr[best]), 5),
        "best_f1_precision": round(float(pr[best]), 4),
        "best_f1_recall": round(float(rc[best]), 4),
    }


def main() -> None:
    splits, _ = load_splits()
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["validation"]
    X_train, X_val = X_train.toarray(), X_val.toarray()   # dense is much faster for distance search
    pd.set_option("display.width", 220, "display.max_columns", None, "display.max_rows", None)

    print("1. Data (training years 2016-2022, validation year 2023; the test set is not loaded into any step)")
    print(f"train {X_train.shape} | KEV {int(y_train.sum()):,} ({y_train.mean() * 100:.3f}%)")
    print(f"val   {X_val.shape} | KEV {int(y_val.sum()):,} ({y_val.mean() * 100:.3f}%)")

    # 2. Majority-class baseline on the validation year
    print("\n2. Majority-class baseline on validation (always predict 'not KEV')")
    print(f"accuracy {(y_val == 0).mean():.5f} | precision 0 | recall 0 | F1 0 "
          f"| average precision of a no-skill model = KEV share = {y_val.mean():.4f}")
    all_pos_f1 = 2 * y_val.mean() / (1 + y_val.mean())
    print(f"Predicting EVERY CVE as KEV instead: precision {y_val.mean():.4f}, recall 1, F1 {all_pos_f1:.4f} "
          "(a best_f1 close to this means no real skill)")

    # 3. One neighbour search for the largest k; smaller k and both weightings are derived from it
    k_max = max(K_VALUES)
    print(f"\n3. Searching the {k_max} nearest training rows for every validation row (a few minutes on the full data)")
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=k_max, algorithm="brute", metric="euclidean", n_jobs=-1).fit(X_train)
    dist, idx = nn.kneighbors(X_val)
    neigh_y = y_train[idx]
    print(f"done in {time.time() - t0:.0f} s")

    # 4. Check my voting against scikit-learn's own KNeighborsClassifier on a validation sample
    rng = np.random.default_rng(42)
    sample = rng.choice(len(y_val), size=min(2000, len(y_val)), replace=False)
    print("\n4. Check against scikit-learn KNeighborsClassifier (k=21, 2,000 validation rows)")
    for w in WEIGHTS:
        ref = KNeighborsClassifier(n_neighbors=21, weights=w, algorithm="brute", n_jobs=-1).fit(X_train, y_train)
        ref_p = ref.predict_proba(X_val[sample])[:, 1]
        mine = vote_probability(neigh_y[sample], dist[sample], 21, w)
        print(f"{w:<9}: max |probability difference| = {np.abs(ref_p - mine).max():.6f} "
              f"| rows with a different 0.5 vote: {int(((ref_p > 0.5) != (mine > 0.5)).sum())}")

    # 5. Grid: k x weighting on the validation year
    rows, probs = [], {}
    for w in WEIGHTS:
        for k in K_VALUES:
            p = vote_probability(neigh_y, dist, k, w)
            probs[(w, k)] = p
            rows.append({"weights": w, "k": k, **evaluate(y_val, p)})
    grid = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    grid.to_csv(REPORTS / "knn_validation_grid.csv", index=False)

    print("\n5. Validation results for every k and voting scheme")
    print("(columns ending _at_0.5 use the plain majority vote; best_f1 uses the best threshold on this same validation year)")
    print(grid.to_string(index=False))

    # 6. Choose by average precision (threshold-free), then set the decision threshold on validation
    best = grid.sort_values("avg_precision", ascending=False).iloc[0]
    print("\n6. Selected configuration (highest validation average precision)")
    print(f"weights={best['weights']} | k={int(best['k'])} | average precision {best['avg_precision']:.4f} "
          f"| ROC AUC {best['roc_auc']:.4f}")
    print(f"Majority vote (threshold 0.5): precision {best['precision_at_0.5']:.4f}, recall {best['recall_at_0.5']:.4f}, "
          f"F1 {best['f1_at_0.5']:.4f}")
    print(f"Best-F1 threshold {best['best_f1_threshold']}: precision {best['best_f1_precision']:.4f}, "
          f"recall {best['best_f1_recall']:.4f}, F1 {best['best_f1']:.4f}")

    # 7. Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for w, color in zip(WEIGHTS, ("#4c72b0", "#c44e52")):
        part = grid[grid["weights"] == w]
        axes[0].plot(part["k"], part["avg_precision"], marker="o", color=color, label=w)
        axes[1].plot(part["k"], part["best_f1"], marker="o", color=color, label=w)
    axes[0].axhline(y_val.mean(), color="grey", linestyle="--", label="no-skill (KEV share)")
    axes[0].set(title="kNN on validation (2023): average precision", xlabel="k (log scale)", ylabel="Average precision",
                xscale="log")
    axes[1].set(title="kNN on validation (2023): best F1 over thresholds", xlabel="k (log scale)", ylabel="F1", xscale="log")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "day3_knn_validation.png", dpi=150)
    plt.close(fig)

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "validation_kev": int(y_val.sum()),
        "k_values": K_VALUES,
        "weights": WEIGHTS,
        "selection_rule": "highest validation average precision",
        "selected": {"weights": str(best["weights"]), "k": int(best["k"]),
                     "threshold_best_f1_on_validation": float(best["best_f1_threshold"])},
        "validation_metrics_selected": {k: (v if isinstance(v, str) else float(v)) for k, v in best.items()},
        "test_set_used": False,
    }
    (META_DIR / "knn_tuning.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved reports/knn_validation_grid.csv, {FIGURES / 'day3_knn_validation.png'} and data/metadata/knn_tuning.json")


if __name__ == "__main__":
    main()
