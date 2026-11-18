"""Inspect the class distribution of is_kev: overall, by status, by year, and time-to-KEV."""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
META_DIR = ROOT / "data" / "metadata"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def summarize(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    pos = int(df["is_kev"].sum())
    neg = n - pos
    print(f"{label}: {n} rows | KEV {pos} ({pos / n * 100:.3f}%) | non-KEV {neg} | ratio {neg / pos:.0f}:1")
    return {"rows": n, "kev": pos, "non_kev": neg, "kev_pct": round(pos / n * 100, 4)}


def rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return (
        df.groupby(by)
        .agg(cves=("cve_id", "size"), kev=("is_kev", "sum"))
        .assign(kev_rate_pct=lambda t: (t["kev"] / t["cves"] * 100).round(3))
    )


def main() -> None:
    df = pd.read_parquet(INTERIM / "nvd_kev_clean.parquet")
    rejected = df["vuln_status"] == "Rejected"
    kept = df[~rejected]
    pd.set_option("display.width", 200, "display.max_rows", None)

    print("1. Class distribution")
    all_info = summarize(df, "All rows         ")
    kept_info = summarize(kept, "Without Rejected ")
    print(f"Rejected rows: {int(rejected.sum())} | of which is_kev=1: {int((rejected & (df['is_kev'] == 1)).sum())}")

    print("\n2. By vuln_status")
    print(rate_table(df, "vuln_status").sort_values("cves", ascending=False).to_string())

    print("\n3. By publication year (Rejected excluded)")
    by_year = rate_table(kept, "published_year")
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(REPORTS / "class_distribution_by_year.csv")
    print(by_year.to_string())

    print("\n4. Time from NVD publication to KEV addition (KEV rows, in days)")
    k = kept[kept["is_kev"] == 1]
    lag = (k["kev_date_added"] - k["published"]).dt.days
    print(lag.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_string())
    within = {d: round(float((lag <= d).mean() * 100), 1) for d in (30, 90, 365)}
    for d, pct in within.items():
        print(f"KEV added within {d} days of publication: {pct}%")
    print(f"KEV added before the NVD publication date (negative lag): {int((lag < 0).sum())}")
    print("\nKEV additions per year:")
    print(k["kev_date_added"].dt.year.value_counts().sort_index().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    years = by_year.index.astype(int)
    axes[0].bar(years, by_year["kev"])
    axes[0].set_title("KEV CVEs by publication year")
    axes[0].set_xlabel("Publication year")
    axes[0].set_ylabel("KEV CVEs")
    axes[1].bar(years, by_year["kev_rate_pct"])
    axes[1].set_title("KEV rate (%) by publication year")
    axes[1].set_xlabel("Publication year")
    axes[1].set_ylabel("% of CVEs in KEV")
    fig.tight_layout()
    fig.savefig(FIGURES / "kev_by_year.png", dpi=150)

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_rows": all_info,
        "without_rejected": kept_info,
        "rejected_rows": int(rejected.sum()),
        "days_to_kev_median": float(lag.median()),
        "pct_kev_added_within_days": within,
    }
    (META_DIR / "class_distribution.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved {FIGURES / 'kev_by_year.png'} and {REPORTS / 'class_distribution_by_year.csv'}")


if __name__ == "__main__":
    main()