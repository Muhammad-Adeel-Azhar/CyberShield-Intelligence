"""Day 3, Part B: the one-time evaluation of the tuned kNN and Decision Tree on the 2024 test set.

Settings and thresholds come from the tuning runs (validation year only). Nothing is changed after seeing test results.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from src.load_data import load_splits
from src.train_knn import K_VALUES, vote_probability

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "metadata"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42
N_BOOT = 1000
BLUE, RED, GREY = "#4c72b0", "#c44e52", "#8c8c8c"
MODEL_ORDER = ["Majority baseline", "kNN", "Decision Tree"]
RULE_DEFAULT = "default rule (kNN: majority vote > 0.5, Tree: predict())"
RULE_TUNED = "validation-tuned threshold"


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold with the best F1 on the given data (same computation as in the tuning scripts)."""
    pr, rc, thr = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(thr[int(f1.argmax())])


def result_row(rule: str, model: str, y: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"decision_rule": rule, "model": model, "accuracy": (tp + tn) / len(y), "precision": prec,
            "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged": tp + fp}


def f1_of(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def show_table(results: list[dict], rule: str, n_kev: int) -> None:
    part = {r["model"]: r for r in results if r["decision_rule"] == rule}
    spec = [("Accuracy", "accuracy", "f"), ("Precision", "precision", "f"), ("Recall", "recall", "f"),
            ("F1-score", "f1", "f"), (f"KEV found (of {n_kev})", "tp", "i"), ("CVEs flagged as KEV", "flagged", "i")]
    table = pd.DataFrame({m: [f"{part[m][k]:.4f}" if t == "f" else f"{part[m][k]:,}" for _, k, t in spec]
                          for m in MODEL_ORDER}, index=[name for name, _, _ in spec])
    print(table.to_string())


def plot_confusion(path: Path, model: str, y: np.ndarray, preds: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (title, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y, pred, labels=[0, 1])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({share[i, j] * 100:.1f}% of row)", ha="center", va="center",
                        color="white" if share[i, j] > 0.5 else "black", fontsize=10)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not KEV", "KEV"], yticklabels=["not KEV", "KEV"],
               xlabel="Predicted", ylabel="Actual", title=title)
        ax.grid(False)
    fig.suptitle(f"{model}: confusion matrices on the 2024 test set (colour = share of each actual class)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    knn_sel = json.loads((META_DIR / "knn_tuning.json").read_text(encoding="utf-8"))["selected"]
    tree_sel = json.loads((META_DIR / "tree_tuning.json").read_text(encoding="utf-8"))["selected"]
    depth = None if tree_sel["max_depth"] == "None" else int(tree_sel["max_depth"])
    class_weight = None if tree_sel["class_weight"] == "none" else tree_sel["class_weight"]
    k, weights = int(knn_sel["k"]), knn_sel["weights"]
    leaf = int(tree_sel["min_samples_leaf"])

    splits, _ = load_splits()
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["validation"]
    X_test, y_test, test_ids = splits["test"]
    n_kev = int(y_test.sum())
    pd.set_option("display.width", 200, "display.max_columns", None)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("1. Settings taken from the validation-year tuning (nothing is changed here)")
    print(f"kNN          : weights={weights}, k={k}")
    print(f"Decision Tree: max_depth={depth}, min_samples_leaf={leaf}, class_weight={class_weight}")
    print(f"Test set (2024): {len(y_test):,} rows | KEV {n_kev} ({y_test.mean() * 100:.3f}%)  <- used once, here")

    # ---- kNN: same search as in tuning (301 neighbours, then the first k), validation first to rebuild the threshold
    t0 = time.time()
    Xtr64 = X_train.toarray()
    nn = NearestNeighbors(n_neighbors=max(K_VALUES), algorithm="brute", metric="euclidean", n_jobs=-1).fit(Xtr64)

    def knn_prob(X_sparse) -> np.ndarray:
        dist, idx = nn.kneighbors(X_sparse.toarray())
        return vote_probability(y_train[idx], dist, k, weights)

    p_knn_val, p_knn = knn_prob(X_val), knn_prob(X_test)

    # ---- Decision Tree: same construction as in tuning
    Xtr32, Xval32, Xte32 = (m.toarray().astype(np.float32) for m in (X_train, X_val, X_test))
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight=class_weight, random_state=SEED)
    tree.fit(Xtr32, y_train)
    p_tree_val, p_tree = tree.predict_proba(Xval32)[:, 1], tree.predict_proba(Xte32)[:, 1]
    print(f"\nModels fitted and scored in {time.time() - t0:.0f} s")

    # Thresholds are rebuilt from the VALIDATION year and compared with the tuning files
    thr_knn, thr_tree = best_threshold(y_val, p_knn_val), best_threshold(y_val, p_tree_val)
    print("\n2. Decision thresholds (best F1 on the validation year, rebuilt here)")
    for name, thr, saved in (("kNN", thr_knn, knn_sel["threshold_best_f1_on_validation"]),
                             ("Decision Tree", thr_tree, tree_sel["threshold_best_f1_on_validation"])):
        status = "matches" if round(thr, 5) == round(saved, 5) else "DIFFERS FROM"
        print(f"{name:<13}: {thr:.5f}  ({status} the tuning run: {saved})")

    assert (p_tree > 0.5).astype(int).tolist() == tree.predict(Xte32).tolist(), "tree default rule mismatch"

    preds = {
        RULE_DEFAULT: {"kNN": p_knn > 0.5, "Decision Tree": p_tree > 0.5},
        RULE_TUNED: {"kNN": p_knn >= thr_knn, "Decision Tree": p_tree >= thr_tree},
    }
    results = []
    for rule, by_model in preds.items():
        results.append(result_row(rule, "Majority baseline", y_test, np.zeros(len(y_test), dtype=bool)))
        for model, pred in by_model.items():
            results.append(result_row(rule, model, y_test, pred))

    prev = y_test.mean()
    print("\n3. Baselines on the test set")
    print(f"Majority class ('always not KEV'): accuracy {1 - prev:.4f}, precision 0, recall 0, F1 0")
    print(f"'Flag every CVE as KEV'          : precision {prev:.4f}, recall 1.0000, F1 {2 * prev / (1 + prev):.4f}")
    print(f"No-skill average precision       : {prev:.4f}")

    print(f"\n4A. Comparison table, {RULE_DEFAULT}")
    show_table(results, RULE_DEFAULT, n_kev)
    print(f"\n4B. Comparison table, {RULE_TUNED}  (kNN {thr_knn:.5f}, Decision Tree {thr_tree:.5f})")
    show_table(results, RULE_TUNED, n_kev)

    # ---- Threshold-free metrics with bootstrap confidence intervals
    ap = {"kNN": average_precision_score(y_test, p_knn), "Decision Tree": average_precision_score(y_test, p_tree)}
    auc = {"kNN": roc_auc_score(y_test, p_knn), "Decision Tree": roc_auc_score(y_test, p_tree)}
    rng = np.random.default_rng(SEED)
    boot = {"ap_knn": [], "ap_tree": [], "f1_knn": [], "f1_tree": []}
    pk_t, pt_t = preds[RULE_TUNED]["kNN"], preds[RULE_TUNED]["Decision Tree"]
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y_test), len(y_test))
        yb = y_test[idx]
        if yb.sum() == 0:
            continue
        boot["ap_knn"].append(average_precision_score(yb, p_knn[idx]))
        boot["ap_tree"].append(average_precision_score(yb, p_tree[idx]))
        boot["f1_knn"].append(f1_of(yb, pk_t[idx]))
        boot["f1_tree"].append(f1_of(yb, pt_t[idx]))
    boot = {key: np.array(v) for key, v in boot.items()}

    def ci(a: np.ndarray) -> str:
        lo, hi = np.percentile(a, [2.5, 97.5])
        return f"[{lo:.4f}, {hi:.4f}]"

    diff = boot["ap_knn"] - boot["ap_tree"]
    print(f"\n5. Threshold-free metrics and uncertainty ({N_BOOT} bootstrap resamples of the test rows, 95% intervals)")
    print(f"{'':<14}{'avg precision':>14}  {'95% interval':<20}{'ROC AUC':>9}  {'F1 (tuned thr.)':>16}  95% interval")
    for model, key in (("kNN", "knn"), ("Decision Tree", "tree")):
        f1_val = [r["f1"] for r in results if r["model"] == model and r["decision_rule"] == RULE_TUNED][0]
        print(f"{model:<14}{ap[model]:>14.4f}  {ci(boot['ap_' + key]):<20}{auc[model]:>9.4f}  {f1_val:>16.4f}  "
              f"{ci(boot['f1_' + key])}")
    print(f"Average precision, kNN minus Decision Tree: {ap['kNN'] - ap['Decision Tree']:+.4f}, 95% interval {ci(diff)}; "
          f"kNN higher in {(diff > 0).mean() * 100:.1f}% of resamples")

    # ---- Confusion matrices and classification reports
    print("\n6. Confusion matrices (rows = actual, columns = predicted; order: not KEV, KEV)")
    for model in ("kNN", "Decision Tree"):
        for rule in (RULE_DEFAULT, RULE_TUNED):
            cm = confusion_matrix(y_test, preds[rule][model], labels=[0, 1])
            print(f"{model} | {rule}: TN {cm[0, 0]:,}  FP {cm[0, 1]:,}  FN {cm[1, 0]:,}  TP {cm[1, 1]:,}")
    report_text = []
    for model in ("kNN", "Decision Tree"):
        for rule in (RULE_DEFAULT, RULE_TUNED):
            rep = classification_report(y_test, preds[rule][model], target_names=["not KEV", "KEV"], digits=4,
                                        zero_division=0)
            report_text.append(f"=== {model} | {rule} ===\n{rep}")
    print("\n7. Classification reports")
    print("\n".join(report_text))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "classification_reports.txt").write_text("\n".join(report_text), encoding="utf-8")

    for model, fname in (("kNN", "day3_confusion_knn.png"), ("Decision Tree", "day3_confusion_tree.png")):
        plot_confusion(FIGURES / fname, model, y_test,
                       {"Default rule": preds[RULE_DEFAULT][model], "Validation-tuned threshold": preds[RULE_TUNED][model]})

    # ---- Decision Tree feature importance
    names = [n.split("__", 1)[-1] for n in json.loads((MODELS / "feature_names.json").read_text(encoding="utf-8"))]
    assert len(names) == len(tree.feature_importances_), "feature names do not match the tree input"
    imp = pd.Series(tree.feature_importances_, index=names).sort_values(ascending=False)
    top = imp.head(20)
    print("\n8. Decision Tree: 15 most important features (share of total impurity reduction)")
    print(imp.head(15).round(4).to_string())
    print(f"Features with non-zero importance: {int((imp > 0).sum())} of {len(imp)}")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh(top.index[::-1], top.values[::-1], color=BLUE)
    ax.set(title="Decision Tree feature importance (top 20)", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(FIGURES / "day3_tree_feature_importance.png", dpi=150)
    plt.close(fig)

    # ---- Model comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, rule, title in ((axes[0], RULE_DEFAULT, "Default decision rules"),
                            (axes[1], RULE_TUNED, "Validation-tuned thresholds")):
        part = {r["model"]: r for r in results if r["decision_rule"] == rule}
        x = np.arange(3)
        for offset, model, color in ((-0.2, "kNN", BLUE), (0.2, "Decision Tree", RED)):
            vals = [part[model][m] for m in ("precision", "recall", "f1")]
            bars = ax.bar(x + offset, vals, width=0.4, color=color, label=model)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=8)
        ax.set(xticks=x, xticklabels=["Precision", "Recall", "F1"], ylim=(0, 1), title=f"{title} (2024 test set)")
        ax.legend()
    for model, p, color in (("kNN", p_knn, BLUE), ("Decision Tree", p_tree, RED)):
        pr, rc, _ = precision_recall_curve(y_test, p)
        axes[2].plot(rc, pr, color=color, label=f"{model} (AP {ap[model]:.3f})")
    axes[2].axhline(prev, color=GREY, linestyle="--", label=f"no-skill ({prev:.3f})")
    axes[2].set(title="Precision-recall curves (2024 test set)", xlabel="Recall", ylabel="Precision", ylim=(0, 1))
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "day3_model_comparison.png", dpi=150)
    plt.close(fig)

    # ---- Save numbers
    pd.DataFrame(results).to_csv(REPORTS / "test_results.csv", index=False)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED / "test_scores.npz", cve_id=test_ids, y_true=y_test, p_knn=p_knn, p_tree=p_tree)
    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_year": 2024, "test_rows": int(len(y_test)), "test_kev": n_kev,
        "knn": {"weights": weights, "k": k, "threshold": thr_knn},
        "tree": {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": class_weight, "threshold": thr_tree},
        "results": results,
        "average_precision": ap, "roc_auc": auc,
        "bootstrap": {"resamples": N_BOOT, "ap_knn_95": np.percentile(boot["ap_knn"], [2.5, 97.5]).tolist(),
                      "ap_tree_95": np.percentile(boot["ap_tree"], [2.5, 97.5]).tolist(),
                      "ap_diff_knn_minus_tree_95": np.percentile(diff, [2.5, 97.5]).tolist()},
        "top_tree_features": imp.head(15).round(5).to_dict(),
    }
    (META_DIR / "test_evaluation.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nSaved reports/test_results.csv, reports/classification_reports.txt, data/metadata/test_evaluation.json,"
          "\ndata/processed/test_scores.npz and four figures: day3_confusion_knn.png, day3_confusion_tree.png,"
          "\nday3_tree_feature_importance.png, day3_model_comparison.png (in reports/figures)")


if __name__ == "__main__":
    main()
