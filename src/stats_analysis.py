"""Day 2, Part A: descriptive statistics of the final dataset."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "nvd_kev_dataset.parquet"
META_DIR = ROOT / "data" / "metadata"
REPORTS = ROOT / "reports"

VERSIONS = {
    "CVSS 2.0": "cvss_v2_score",
    "CVSS 3.0": "cvss_v30_score",
    "CVSS 3.1": "cvss_v31_score",
    "CVSS 4.0": "cvss_v40_score",
}
# Preference order for the single "best available" score
BEST_ORDER = [
    ("cvss_v31_score", "v3.1"),
    ("cvss_v30_score", "v3.0"),
    ("cvss_v40_score", "v4.0"),
    ("cvss_v2_score", "v2.0"),
]
KEV_ONLY_COLUMNS = ["kev_date_added", "nvd_cisa_exploit_add"]


def best_available(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    score = df[BEST_ORDER[0][0]].copy()
    for col, _ in BEST_ORDER[1:]:
        score = score.combine_first(df[col])
    source = np.select([df[c].notna() for c, _ in BEST_ORDER], [lab for _, lab in BEST_ORDER], default="none")
    return score, pd.Series(source, index=df.index)


def score_stats(s: pd.Series, total: int) -> dict:
    return {
        "count": int(s.count()),
        "pct_of_all": round(s.count() / total * 100, 2),
        "mean": round(float(s.mean()), 3),
        "median": round(float(s.median()), 3),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": round(float(s.std()), 3),
    }


def main() -> None:
    df = pd.read_parquet(DATA)
    n = len(df)
    kev = df["is_kev"] == 1
    n_kev = int(kev.sum())
    best, source = best_available(df)
    pd.set_option("display.width", 200, "display.max_columns", None, "display.max_rows", None)
    REPORTS.mkdir(exist_ok=True)

    # 1. Counts
    print("1. Counts")
    print(f"Total vulnerabilities : {n:,}")
    print(f"KEV records           : {n_kev:,} ({n_kev / n * 100:.3f}%)")
    print(f"Non-KEV records       : {n - n_kev:,} ({(n - n_kev) / n * 100:.3f}%)")
    print(f"Class ratio           : {(n - n_kev) / n_kev:.0f}:1")

    # 2. CVSS statistics
    rows = {name: score_stats(df[col], n) for name, col in VERSIONS.items()}
    rows["Best available"] = score_stats(best, n)
    stats = pd.DataFrame(rows).T
    stats.to_csv(REPORTS / "day2_cvss_statistics.csv")
    print("\n2. CVSS base score statistics (rows without that version are not counted)")
    print(stats.to_string())

    split = pd.DataFrame({
        "KEV": score_stats(best[kev], n_kev),
        "non-KEV": score_stats(best[~kev], n - n_kev),
    }).T[["count", "mean", "median", "min", "max", "std"]]
    print("\nBest available score, KEV vs non-KEV")
    print(split.to_string())
    print("\nWhich version supplied the best available score:")
    print(source.value_counts().to_string())

    # 3. Missing values
    miss = pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "missing_pct_kev": (df[kev].isna().mean() * 100).round(2),
        "missing_pct_non_kev": (df[~kev].isna().mean() * 100).round(2),
    }).sort_values("missing_pct", ascending=False)
    miss.to_csv(REPORTS / "day2_missing_values.csv")
    print(f"\n3. Missing values (top 14 of {len(miss)} columns; full table in reports/day2_missing_values.csv)")
    print(f"   {KEV_ONLY_COLUMNS} are empty for non-KEV rows by design")
    print(miss.head(14).to_string())

    # 4. Duplicates
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df.duplicated("cve_id").sum())
    dup_desc = int(df.duplicated("description").sum())
    print("\n4. Duplicates")
    print(f"Fully duplicated rows        : {dup_rows}")
    print(f"Duplicated CVE IDs           : {dup_ids}")
    print(f"Repeated descriptions (info) : {dup_desc}")

    # 5. Per publication year
    year = df.groupby("published_year").agg(cves=("cve_id", "size"), kev=("is_kev", "sum"))
    year["kev_rate_pct"] = (year["kev"] / year["cves"] * 100).round(3)
    year["mean_best_score"] = best.groupby(df["published_year"]).mean().round(2)
    year.to_csv(REPORTS / "day2_vulns_per_year.csv")
    print("\n5. Vulnerabilities per publication year (last 15; all years in reports/day2_vulns_per_year.csv)")
    print(year.tail(15).to_string())
    print(f"Years covered: {int(year.index.min())} to {int(year.index.max())} ({len(year)} years)")

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "total": n,
        "kev": n_kev,
        "non_kev": n - n_kev,
        "cvss_statistics": rows,
        "best_score_by_class": {
            "kev": score_stats(best[kev], n_kev),
            "non_kev": score_stats(best[~kev], n - n_kev),
        },
        "best_score_source_counts": source.value_counts().to_dict(),
        "duplicate_rows": dup_rows,
        "duplicate_cve_ids": dup_ids,
        "repeated_descriptions": dup_desc,
        "columns_over_20pct_missing": miss.index[miss["missing_pct"] > 20].tolist(),
    }
    (META_DIR / "day2_statistics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nSaved reports/day2_*.csv and data/metadata/day2_statistics.json")


if __name__ == "__main__":
    main()