"""Day 3, Part C, Algorithm 2: DBSCAN on the same 20,000-row sample and the same features as K-Means. is_kev is NEVER an input.

Stage 1:  python -m src.train_dbscan
          -> k-distance diagnostics and an eps x min_samples grid (labels are not used at all)
Stage 2:  python -m src.train_dbscan --eps 1.2 --min-samples 10
          -> final DBSCAN with the chosen settings: clusters, noise, ARI/NMI, PCA plot
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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from src.train_kmeans import FIGURES, META_DIR, N_PERMUTATIONS, REPORTS, SEED, load_sample, profile_clusters

MIN_SAMPLES_GRID = [5, 10, 20, 50]
EPS_PERCENTILES = [50, 70, 85, 95]   # eps = this percentile of the k-distance, i.e. about that % of rows become core points
MAX_PAIRS = 80_000_000               # skip settings whose neighbourhoods would not fit in memory


def estimated_pairs(X: np.ndarray, eps: float) -> float:
    """Estimate the total number of neighbour pairs from 500 probe rows (cheap safety check before running DBSCAN)."""
    rng = np.random.default_rng(SEED)
    probe = rng.choice(len(X), size=min(500, len(X)), replace=False)
    nn = NearestNeighbors(radius=eps, algorithm="brute", n_jobs=-1).fit(X)
    counts = [len(a) for a in nn.radius_neighbors(X[probe], return_distance=False)]
    return float(np.mean(counts)) * len(X)


def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> DBSCAN:
    return DBSCAN(eps=eps, min_samples=min_samples, algorithm="brute", n_jobs=-1).fit(X)


def stage_grid(X: np.ndarray) -> None:
    pd.set_option("display.width", 200, "display.max_columns", None)
    m_max = max(MIN_SAMPLES_GRID)
    dist, _ = NearestNeighbors(n_neighbors=m_max, algorithm="brute", n_jobs=-1).fit(X).kneighbors(X)
    kd = {m: dist[:, m - 1] for m in MIN_SAMPLES_GRID}   # distance to the m-th nearest row (the row itself counts as 1st)

    print("1. k-distance: distance from each row to its m-th nearest row (m = min_samples), percentiles over all rows")
    pct = [10, 25, 50, 75, 90, 95, 99]
    table = pd.DataFrame({m: np.percentile(kd[m], pct) for m in MIN_SAMPLES_GRID}, index=[f"p{p}" for p in pct]).T
    table["rows_with_distance_0_pct"] = [round(float((kd[m] == 0).mean() * 100), 2) for m in MIN_SAMPLES_GRID]
    table.index.name = "min_samples"
    print(table.round(3).to_string())
    print("(distance 0 means m identical rows exist; eps must be above the typical k-distance for a row to be a core point)")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MIN_SAMPLES_GRID:
        ax.plot(np.sort(kd[m]), label=f"min_samples={m}")
    ax.set(title="Sorted k-distance curves (the knee suggests eps)", xlabel="Rows sorted by k-distance",
           ylabel="Distance to the m-th nearest row", ylim=(0, float(np.percentile(kd[m_max], 99.5))))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "day3_dbscan_kdistance.png", dpi=150)
    plt.close(fig)

    print("\n2. Grid: each eps is a percentile of that min_samples' k-distance (labels are not used)")
    rows = []
    for m in MIN_SAMPLES_GRID:
        for p in EPS_PERCENTILES:
            eps = round(float(np.percentile(kd[m], p)), 3)
            if eps <= 0:
                continue
            pairs = estimated_pairs(X, eps)
            row = {"min_samples": m, "eps": eps, "approx_core_pct": p, "neighbours_per_row": round(pairs / len(X), 1)}
            if pairs > MAX_PAIRS:
                row.update({"clusters": None, "noise_pct": None, "largest_cluster_pct": None, "note": "skipped: too large"})
            else:
                labels = run_dbscan(X, eps, m).labels_
                clustered = labels[labels >= 0]
                row.update({
                    "clusters": int(len(set(clustered))),
                    "noise_pct": round(float((labels == -1).mean() * 100), 2),
                    "largest_cluster_pct": round(float(np.bincount(clustered).max() / len(X) * 100), 2) if len(clustered) else 0.0,
                    "note": "",
                })
            rows.append(row)
            print(f"min_samples={m:<3} eps={eps:<8} -> clusters {row['clusters']} | noise {row['noise_pct']}%")
    grid = pd.DataFrame(rows).drop_duplicates(subset=["min_samples", "eps"])
    grid.to_csv(REPORTS / "dbscan_grid.csv", index=False)
    print("\nGrid summary")
    print(grid.to_string(index=False))
    print(f"\nSaved reports/dbscan_grid.csv and {FIGURES / 'day3_dbscan_kdistance.png'}")
    print("Pick a setting with a sensible number of clusters, moderate noise and no single giant cluster, then run:")
    print("python -m src.train_dbscan --eps VALUE --min-samples VALUE")


def stage_final(X: np.ndarray, y: np.ndarray, ids: np.ndarray, eps: float, m: int) -> None:
    pd.set_option("display.width", 220, "display.max_columns", None)
    pairs = estimated_pairs(X, eps)
    if pairs > MAX_PAIRS:
        raise SystemExit(f"eps={eps} gives about {pairs / len(X):,.0f} neighbours per row: too large for memory. Use a smaller eps.")
    db = run_dbscan(X, eps, m)
    labels = db.labels_
    noise = labels == -1
    n_clusters = int(len(set(labels[~noise])))
    print(f"DBSCAN eps={eps}, min_samples={m}")
    print(f"Clusters found: {n_clusters} | core points: {len(db.core_sample_indices_):,} "
          f"({len(db.core_sample_indices_) / len(X) * 100:.1f}%) | noise points: {int(noise.sum()):,} ({noise.mean() * 100:.1f}%)")

    print("\nGroups (is_kev was NOT used to build them; it is shown here only to evaluate them afterwards)")
    print(f"Sample KEV rate: {y.mean() * 100:.3f}% ({int(y.sum())} of {len(y):,}) | cluster -1 = noise")
    table = profile_clusters(labels, y, ids).sort_values("rows", ascending=False)
    table.to_csv(REPORTS / "dbscan_clusters.csv")
    shown = table.head(15)
    if -1 in table.index and -1 not in shown.index:
        shown = pd.concat([shown, table.loc[[-1]]])
    print(shown.to_string())
    if len(table) > len(shown):
        print(f"({len(table) - len(shown)} smaller groups not shown; all are in reports/dbscan_clusters.csv)")

    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    rng = np.random.default_rng(SEED)
    null_ari, null_nmi = [], []
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(y)
        null_ari.append(adjusted_rand_score(shuffled, labels))
        null_nmi.append(normalized_mutual_info_score(shuffled, labels))
    print("\nAgreement with the real KEV labels (noise counted as one extra group)")
    print(f"ARI {ari:.5f} | NMI {nmi:.5f}")
    print(f"Chance level ({N_PERMUTATIONS} random shuffles): ARI mean {np.mean(null_ari):.5f}, 95th percentile "
          f"{np.percentile(null_ari, 95):.5f} | NMI mean {np.mean(null_nmi):.5f}, 95th percentile "
          f"{np.percentile(null_nmi, 95):.5f}")
    ari_c = nmi_c = None
    if (~noise).sum() > 1 and len(set(labels[~noise])) > 1:
        ari_c = adjusted_rand_score(y[~noise], labels[~noise])
        nmi_c = normalized_mutual_info_score(y[~noise], labels[~noise])
        print(f"Clustered points only (noise removed, {int((~noise).sum()):,} rows, {int(y[~noise].sum())} KEV): "
              f"ARI {ari_c:.5f} | NMI {nmi_c:.5f}")
    print(f"KEV among noise points: {int(y[noise].sum())} of {int(y.sum())} "
          f"({(y[noise].mean() * 100 if noise.any() else 0):.3f}% of noise rows vs {y.mean() * 100:.3f}% overall)")

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    Z = pca.transform(X)
    print(f"\nPCA: the two plotted components hold {pca.explained_variance_ratio_.sum() * 100:.1f}% of the variance")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(Z[noise, 0], Z[noise, 1], s=3, alpha=0.3, color="lightgrey", label=f"noise ({int(noise.sum()):,})")
    sizes = pd.Series(labels[~noise]).value_counts()
    top = list(sizes.index[:9])
    cmap = plt.get_cmap("tab10")
    for rank, lab in enumerate(top):
        mk = labels == lab
        ax.scatter(Z[mk, 0], Z[mk, 1], s=5, alpha=0.6, color=cmap(rank), label=f"cluster {lab} ({int(mk.sum()):,})")
    other = ~noise & ~np.isin(labels, top)
    if other.any():
        ax.scatter(Z[other, 0], Z[other, 1], s=5, alpha=0.6, color="black", label=f"other clusters ({int(other.sum()):,})")
    ax.set(title=f"DBSCAN (eps={eps}, min_samples={m}) on the first two principal components", xlabel="PC 1", ylabel="PC 2")
    ax.legend(fontsize=8, markerscale=3)
    non = y == 0
    axes[1].scatter(Z[non, 0], Z[non, 1], s=4, alpha=0.25, color="#4c72b0", label=f"non-KEV ({int(non.sum()):,})")
    axes[1].scatter(Z[~non, 0], Z[~non, 1], s=14, alpha=0.8, color="#c44e52", label=f"KEV ({int((~non).sum())})")
    axes[1].set(title="Same view coloured by the real label (not used for clustering)", xlabel="PC 1", ylabel="PC 2")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "day3_dbscan_pca.png", dpi=150)
    plt.close(fig)

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rows": int(len(y)), "sample_kev": int(y.sum()), "seed": SEED, "eps": eps, "min_samples": m,
        "clusters": n_clusters, "core_points": int(len(db.core_sample_indices_)), "noise_points": int(noise.sum()),
        "ari": float(ari), "nmi": float(nmi),
        "ari_clustered_only": None if ari_c is None else float(ari_c),
        "nmi_clustered_only": None if nmi_c is None else float(nmi_c),
        "ari_chance_p95": float(np.percentile(null_ari, 95)), "nmi_chance_p95": float(np.percentile(null_nmi, 95)),
        "pca_variance_explained_2d": float(pca.explained_variance_ratio_.sum()),
        "is_kev_used_as_input": False,
    }
    (META_DIR / "dbscan_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved reports/dbscan_clusters.csv, {FIGURES / 'day3_dbscan_pca.png'} and data/metadata/dbscan_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="DBSCAN clustering of CVE features")
    parser.add_argument("--eps", type=float, help="neighbourhood radius for the final run")
    parser.add_argument("--min-samples", type=int, dest="min_samples", help="min_samples for the final run")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y, ids = load_sample()
    print(f"Sample: {X.shape[0]:,} random training rows (2016-2022) x {X.shape[1]} features, the same rows as K-Means\n")
    if args.eps is None and args.min_samples is None:
        stage_grid(X)
    elif args.eps is None or args.min_samples is None:
        raise SystemExit("Give both --eps and --min-samples, or neither.")
    else:
        stage_final(X, y, ids, args.eps, args.min_samples)


if __name__ == "__main__":
    main()
