"""Inspect missing values in the cleaned dataset."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"

SCORE_COLS = ["cvss_v2_score", "cvss_v30_score", "cvss_v31_score", "cvss_v40_score"]


def blank_count(s: pd.Series) -> int:
    """Count empty or whitespace-only strings (not caught by isna)."""
    try:
        return int(s.str.strip().eq("").sum())
    except AttributeError:
        return 0


def main() -> None:
    df = pd.read_parquet(INTERIM / "nvd_kev_clean.parquet")
    kev = df["is_kev"] == 1
    miss = df.isna()

    report = pd.DataFrame(
        {
            "missing": miss.sum(),
            "missing_pct": (miss.mean() * 100).round(2),
            "missing_pct_kev": (miss[kev].mean() * 100).round(2),
            "missing_pct_non_kev": (miss[~kev].mean() * 100).round(2),
            "blank_strings": pd.Series({c: blank_count(df[c]) for c in df.columns}),
        }
    ).sort_values("missing_pct", ascending=False)
    REPORTS.mkdir(exist_ok=True)
    report.to_csv(REPORTS / "missing_values.csv")

    pd.set_option("display.max_rows", None, "display.width", 200)
    print(f"Rows: {len(df)} | columns: {df.shape[1]} | is_kev=1: {int(kev.sum())}\n")
    print("Missing values per column")
    print("(kev_date_added and nvd_cisa_exploit_add are empty by design for non-KEV rows)")
    print(report.to_string())

    no_cvss = df[SCORE_COLS].isna().all(axis=1)
    print(f"\nRows with no CVSS score of any version: {int(no_cvss.sum())} ({no_cvss.mean() * 100:.2f}%)")
    print(f"  of which is_kev=1: {int((no_cvss & kev).sum())}")

    print("\nvuln_status counts:")
    print(df["vuln_status"].value_counts(dropna=False).to_string())

    rejected = df["description"].str.startswith("** REJECT **", na=False)
    print(f"\nDescriptions marked as rejected: {int(rejected.sum())}")

    by_year = (
        df.assign(
            no_cvss=no_cvss,
            no_cpe=df["n_cpe_matches"].eq(0),
            no_cwe=df["cwe_ids"].isna(),
        )
        .groupby("published_year")[["no_cvss", "no_cpe", "no_cwe"]]
        .mean()
        .mul(100)
        .round(1)
    )
    print("\n% of CVEs missing CVSS / CPE matches / CWE, by publication year (last 8 years):")
    print(by_year.tail(8).to_string())
    print(f"\nSaved {REPORTS / 'missing_values.csv'}")


if __name__ == "__main__":
    main()