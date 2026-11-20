"""Day 3, Part C, Algorithm 1: K-Means on the preprocessed features. is_kev is NEVER an input.

Stage 1:  python -m src.train_kmeans            -> elbow + silhouette for k = 2..15 (labels are not used at all)
Stage 2:  python -m src.train_kmeans --k 6      -> final K-Means with the chosen k, cluster profiles, ARI/NMI, PCA plot
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from src.load_data import load_splits

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "metadata"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DATA = ROOT / "data" / "processed" / "model_dataset.parquet"

SAMPLE_SIZE = 20_000      # random rows of the training years (2016-2022); the same rows are reused for DBSCAN
SEED = 42
K_RANGE = range(2, 16)
N_PERMUTATIONS = 200


def load_sample():
    """Random sample of the training years. X holds the preprocessed features only; y is for evaluation afterwards."""
    splits, _ = load_splits()
    X, y, ids = splits["train"]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(X.shape[0], size=min(SAMPLE_SIZE, X.shape[0]), replace=False))
    return X[idx].toarray(), y[idx], ids[idx]


def stage_elbow(X: np.ndarray) -> None:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=5000, random_state=SEED)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"k={k:<2} inertia {km.inertia_:,.0f} | silhouette {sil:.4f}")
    table = pd.DataFrame(rows)
    table["inertia_drop_pct"] = (-table["inertia"].pct_change() * 100).round(2)
    table["inertia"] = table["inertia"].round(0).astype(int)
    table["silhouette"] = table["silhouette"].round(4)
    table.to_csv(REPORTS / "kmeans_elbow.csv", index=False)

    print("\nElbow table (inertia_drop_pct = how much the within-cluster distance falls when k goes up by one)")
    print(table.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(table["k"], table["inertia"], marker="o", color="#4c72b0")
    axes[0].set(title="Elbow: within-cluster sum of squares", xlabel="Number of clusters k", ylabel="Inertia")
    axes[1].plot(table["k"], table["silhouette"], marker="o", color="#c44e52")
    axes[1].set(title="Silhouette score (5,000-row subsample)", xlabel="Number of clusters k", ylabel="Silhouette")
    for ax in axes:
        ax.set_xticks(list(K_RANGE))
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "day3_kmeans_elbow.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved reports/kmeans_elbow.csv and {FIGURES / 'day3_kmeans_elbow.png'}")
    print("Look for the k where the curve bends and the drops become small. Then run:  python -m src.train_kmeans --k N")


def profile_clusters(labels: np.ndarray, y: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    """One row per cluster: size, KEV count and rate (labels used only here, afterwards), and a plain-language profile."""
    cols = ["cve_id", "cvss_base_score", "desc_words", "published_year", "attack_vector", "confidentiality_impact"]
    orig = pd.read_parquet(DATA, columns=cols).set_index("cve_id").reindex(ids).reset_index(drop=True)
    d = pd.DataFrame({
        "cluster": labels, "is_kev": y,
        "cvss": orig["cvss_base_score"], "desc_words": orig["desc_words"], "year": orig["published_year"],
        "network": (orig["attack_vector"] == "NETWORK").astype(float),
        "conf_high": (orig["confidentiality_impact"] == "HIGH").astype(float),
    })
    g = d.groupby("cluster").agg(rows=("is_kev", "size"), kev=("is_kev", "sum"), kev_rate_pct=("is_kev", "mean"),
                                 mean_cvss=("cvss", "mean"), mean_desc_words=("desc_words", "mean"),
                                 mean_year=("year", "mean"), pct_network=("network", "mean"),
                                 pct_conf_high=("conf_high", "mean"))
    base = y.mean()
    g["kev_rate_pct"] *= 100
    g["lift_vs_overall"] = g["kev_rate_pct"] / (base * 100)
    g["pct_network"] *= 100
    g["pct_conf_high"] *= 100
    return g.round(3)


def stage_final(X: np.ndarray, y: np.ndarray, ids: np.ndarray, k: int) -> None:
    pd.set_option("display.width", 220, "display.max_columns", None)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    labels = km.labels_
    sil = silhouette_score(X, labels, sample_size=5000, random_state=SEED)
    print(f"K-Means k={k} | inertia {km.inertia_:,.0f} | silhouette {sil:.4f} | iterations {km.n_iter_}")

    print("\nClusters (is_kev was NOT used to build them; it is shown here only to evaluate them afterwards)")
    print(f"Sample KEV rate: {y.mean() * 100:.3f}% ({int(y.sum())} of {len(y):,})")
    table = profile_clusters(labels, y, ids)
    table.to_csv(REPORTS / "kmeans_clusters.csv")
    print(table.to_string())

    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    rng = np.random.default_rng(SEED)
    null_ari, null_nmi = [], []
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(y)
        null_ari.append(adjusted_rand_score(shuffled, labels))
        null_nmi.append(normalized_mutual_info_score(shuffled, labels))
    print("\nAgreement with the real KEV labels")
    print(f"ARI {ari:.5f} | NMI {nmi:.5f}")
    print(f"Chance level ({N_PERMUTATIONS} random shuffles of the labels): ARI mean {np.mean(null_ari):.5f}, "
          f"95th percentile {np.percentile(null_ari, 95):.5f} | NMI mean {np.mean(null_nmi):.5f}, "
          f"95th percentile {np.percentile(null_nmi, 95):.5f}")

    # PCA to 2D for the picture only (the clustering itself used all features)
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    Z = pca.transform(X)
    centers = pca.transform(km.cluster_centers_)
    print(f"\nPCA: the two plotted components hold {pca.explained_variance_ratio_.sum() * 100:.1f}% of the variance "
          f"({pca.explained_variance_ratio_[0] * 100:.1f}% + {pca.explained_variance_ratio_[1] * 100:.1f}%)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc = axes[0].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10" if k <= 10 else "tab20", s=4, alpha=0.5)
    axes[0].scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=120, edgecolor="white", label="centroids")
    axes[0].set(title=f"K-Means clusters (k={k}) on the first two principal components",
                xlabel="PC 1", ylabel="PC 2")
    axes[0].legend(*sc.legend_elements(), title="cluster", loc="best", fontsize=8)
    non = y == 0
    axes[1].scatter(Z[non, 0], Z[non, 1], s=4, alpha=0.25, color="#4c72b0", label=f"non-KEV ({int(non.sum()):,})")
    axes[1].scatter(Z[~non, 0], Z[~non, 1], s=14, alpha=0.8, color="#c44e52", label=f"KEV ({int((~non).sum())})")
    axes[1].set(title="Same view coloured by the real label (not used for clustering)", xlabel="PC 1", ylabel="PC 2")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "day3_kmeans_pca.png", dpi=150)
    plt.close(fig)

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rows": int(len(y)), "sample_kev": int(y.sum()), "seed": SEED, "k": k,
        "inertia": float(km.inertia_), "silhouette": float(sil),
        "ari": float(ari), "nmi": float(nmi),
        "ari_chance_mean": float(np.mean(null_ari)), "ari_chance_p95": float(np.percentile(null_ari, 95)),
        "nmi_chance_mean": float(np.mean(null_nmi)), "nmi_chance_p95": float(np.percentile(null_nmi, 95)),
        "pca_variance_explained_2d": float(pca.explained_variance_ratio_.sum()),
        "is_kev_used_as_input": False,
    }
    (META_DIR / "kmeans_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved reports/kmeans_clusters.csv, {FIGURES / 'day3_kmeans_pca.png'} and data/metadata/kmeans_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Means clustering of CVE features")
    parser.add_argument("--k", type=int, help="number of clusters for the final fit; leave out to run the elbow stage")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y, ids = load_sample()
    print(f"Sample: {X.shape[0]:,} random training rows (2016-2022) x {X.shape[1]} features "
          f"(numeric standardized, categorical one-hot, as saved by the preprocessor)")
    if args.k is None:
        print("Stage 1: choose k without looking at the labels\n")
        stage_elbow(X)
    else:
        print(f"Stage 2: final fit with k={args.k}\n")
        stage_final(X, y, ids, args.k)


if __name__ == "__main__":
    main()
