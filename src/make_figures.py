"""Day 2, Part B: six visualizations of the final dataset."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "nvd_kev_dataset.parquet"
FIG = ROOT / "reports" / "figures"

BLUE, RED = "#4c72b0", "#c44e52"
PALETTE = {"Non-KEV": BLUE, "KEV": RED}
ORDER = ["Non-KEV", "KEV"]
BEST_ORDER = ["cvss_v31_score", "cvss_v30_score", "cvss_v40_score", "cvss_v2_score"]

sns.set_theme(style="whitegrid", context="notebook")


def save(fig, name: str) -> None:
    fig.tight_layout()
    path = FIG / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    df["class"] = np.where(df["is_kev"] == 1, "KEV", "Non-KEV")
    best = df[BEST_ORDER[0]]
    for col in BEST_ORDER[1:]:
        best = best.combine_first(df[col])
    df["cvss_best"] = best
    n, n_kev = len(df), int((df["is_kev"] == 1).sum())
    print(f"Rows: {n:,} | KEV: {n_kev:,}\n")

    # 1. CVSS distribution -------------------------------------------------
    bins = np.arange(0, 10.5, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.histplot(data=df, x="cvss_best", bins=bins, color=BLUE, ax=axes[0])
    axes[0].set(title="CVSS base score distribution (all CVEs)", xlabel="CVSS base score (best available version)",
                ylabel="Number of CVEs")
    sns.histplot(data=df, x="cvss_best", hue="class", hue_order=ORDER, palette=PALETTE, bins=bins,
                 stat="density", common_norm=False, element="step", ax=axes[1])
    axes[1].set(title="Score distribution within each class (each class sums to 1)",
                xlabel="CVSS base score (best available version)", ylabel="Density")
    save(fig, "day2_01_cvss_distribution.png")

    # 2. Severity boxplot --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, title in (
        (axes[0], "cvss_best", "Best available score"),
        (axes[1], "cvss_v31_score", "CVSS 3.1 score only"),
    ):
        sns.boxplot(data=df, x="class", y=col, order=ORDER, hue="class", palette=PALETTE,
                    legend=False, fliersize=1.5, ax=ax)
        counts = df.groupby("class")[col].count()
        ax.set(title=f"{title}\n(n: non-KEV {counts['Non-KEV']:,}, KEV {counts['KEV']:,})",
               xlabel="", ylabel="CVSS base score")
    save(fig, "day2_02_severity_boxplot.png")
    print("Median score by class:")
    print(df.groupby("class")[["cvss_best", "cvss_v31_score"]].median().round(2).to_string(), "\n")

    # 3. Attack vector countplot ------------------------------------------
    av = df["v3_attack_vector"].fillna("No CVSS 3.x data").rename("attack_vector")
    order = av.value_counts().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x=av, order=order, color=BLUE, ax=axes[0])
    for c in axes[0].containers:
        axes[0].bar_label(c, labels=[f"{int(v):,}" for v in c.datavalues], fontsize=8)
    axes[0].set(title="CVEs by CVSS 3.x attack vector", xlabel="", ylabel="Number of CVEs")
    share = (pd.crosstab(av, df["class"], normalize="columns") * 100).reindex(order)
    long = share.reset_index().melt(id_vars="attack_vector", var_name="class", value_name="percent")
    sns.barplot(data=long, x="attack_vector", y="percent", hue="class", hue_order=ORDER, palette=PALETTE, ax=axes[1])
    axes[1].set(title="Share of each class by attack vector (%)", xlabel="", ylabel="% of the class")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
    save(fig, "day2_03_attack_vector.png")
    table = pd.crosstab(av, df["class"]).reindex(order)
    table["kev_rate_pct"] = (table["KEV"] / (table["KEV"] + table["Non-KEV"]) * 100).round(3)
    print("Attack vector (counts and KEV rate):")
    print(table.to_string(), "\n")

    # 4. Correlation heatmap ----------------------------------------------
    df["desc_words"] = df["description"].str.split().str.len()
    df["desc_chars"] = df["description"].str.len()
    cols = {
        "cvss_best": "CVSS score (best)",
        "v3_exploitability_score": "Exploitability (3.x)",
        "v3_impact_score": "Impact (3.x)",
        "n_cpe_matches": "CPE matches",
        "n_references": "References",
        "published_year": "Publication year",
        "desc_words": "Description words",
        "desc_chars": "Description chars",
        "is_kev": "is_kev (target)",
    }
    corr = df[list(cols)].astype(float).corr(method="spearman").rename(index=cols, columns=cols)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"label": "Spearman correlation"}, ax=ax)
    ax.set_title("Spearman correlation between numerical features")
    ax.grid(False)
    save(fig, "day2_04_correlation_heatmap.png")
    print("Correlation with is_kev (Spearman):")
    print(corr["is_kev (target)"].drop("is_kev (target)").sort_values(ascending=False).round(3).to_string(), "\n")

    # 5. Publication trends ------------------------------------------------
    year = df.groupby("published_year").agg(cves=("cve_id", "size"), kev=("is_kev", "sum"))
    year["rate"] = year["kev"] / year["cves"] * 100
    yrs = year.index.astype(int)
    last = df["published"].max()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(yrs, year["cves"], color=BLUE)
    axes[0].set(title=f"CVEs published per year (2026 only up to {last:%d %b})", xlabel="Publication year",
                ylabel="Number of CVEs")
    axes[1].bar(yrs, year["kev"], color=RED, label="KEV CVEs (left axis)")
    axes[1].set(title="KEV CVEs and KEV rate by publication year", xlabel="Publication year", ylabel="KEV CVEs",
                ylim=(0, year["kev"].max() * 1.3))
    ax2 = axes[1].twinx()
    ax2.plot(yrs, year["rate"], color="black", marker="o", markersize=3, linewidth=1.2, label="KEV rate % (right axis)")
    ax2.set_ylabel("% of that year's CVEs in KEV")
    ax2.grid(False)
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="upper left")
    save(fig, "day2_05_publication_trends.png")

    # 6. Feature scatterplot ----------------------------------------------
    sub = df.dropna(subset=["v3_exploitability_score", "v3_impact_score"])
    kev_pts = sub[sub["class"] == "KEV"]
    non_pts = sub[sub["class"] == "Non-KEV"].sample(n=min(20000, int((sub["class"] == "Non-KEV").sum())), random_state=42)
    rng = np.random.default_rng(42)

    def jitter(s: pd.Series) -> np.ndarray:
        return s.to_numpy(dtype=float) + rng.uniform(-0.04, 0.04, len(s))

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(jitter(non_pts["v3_exploitability_score"]), jitter(non_pts["v3_impact_score"]), s=8, alpha=0.15,
               color=BLUE, label=f"Non-KEV (random sample of {len(non_pts):,})", zorder=1)
    ax.scatter(jitter(kev_pts["v3_exploitability_score"]), jitter(kev_pts["v3_impact_score"]), s=14, alpha=0.6,
               color=RED, label=f"KEV (all {len(kev_pts):,} with CVSS 3.x)", zorder=2)
    ax.set(title="CVSS 3.x exploitability vs impact sub-scores\n(small random jitter added so overlapping points are visible)",
           xlabel="Exploitability sub-score", ylabel="Impact sub-score")
    ax.legend(loc="lower right")
    save(fig, "day2_06_feature_scatter.png")
    print(f"\nScatter: {len(kev_pts):,} KEV and {len(non_pts):,} sampled non-KEV points with CVSS 3.x sub-scores")


if __name__ == "__main__":
    main()