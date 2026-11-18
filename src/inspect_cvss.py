"""Inspect CVSS versions in the cleaned dataset."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"

VERSIONS = {"v2": "CVSS 2.0", "v30": "CVSS 3.0", "v31": "CVSS 3.1", "v40": "CVSS 4.0"}
LABELS = {"v2": "v2", "v30": "v3.0", "v31": "v3.1", "v40": "v4.0"}
PREFIX = {"v30": "CVSS:3.0/", "v31": "CVSS:3.1/", "v40": "CVSS:4.0/"}


def main() -> None:
    df = pd.read_parquet(INTERIM / "nvd_kev_clean.parquet")
    kev = df["is_kev"] == 1
    n_kev = int(kev.sum())
    pd.set_option("display.width", 200, "display.max_columns", None)

    present = pd.DataFrame({tag: df[f"cvss_{tag}_score"].notna() for tag in VERSIONS})

    # 1. Coverage per version
    rows = []
    for tag, name in VERSIONS.items():
        has = present[tag]
        rows.append(
            {
                "version": name,
                "cves_with_score": int(has.sum()),
                "pct_of_all": round(has.mean() * 100, 2),
                "kev_with_score": int((has & kev).sum()),
                "pct_of_kev": round((has & kev).sum() / n_kev * 100, 2),
                "kev_rate_pct": round((has & kev).sum() / has.sum() * 100, 3),
            }
        )
    print("1. Coverage by CVSS version")
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. Combinations of versions per CVE
    pattern = pd.Series("", index=df.index)
    for tag, label in LABELS.items():
        pattern = pattern + present[tag].map({True: label + "+", False: ""})
    df["cvss_versions"] = pattern.str.rstrip("+").replace("", "none")

    patterns = (
        df.groupby("cvss_versions")
        .agg(cves=("cve_id", "size"), kev=("is_kev", "sum"))
        .assign(kev_rate_pct=lambda t: (t["kev"] / t["cves"] * 100).round(3))
        .sort_values("cves", ascending=False)
    )
    REPORTS.mkdir(exist_ok=True)
    patterns.to_csv(REPORTS / "cvss_version_patterns.csv")
    print("\n2. Combinations of CVSS versions per CVE")
    print(patterns.to_string())

    # 3. Sanity checks
    print("\n3. Sanity checks")
    for tag, name in VERSIONS.items():
        s = df[f"cvss_{tag}_score"]
        line = f"{name}: score range {s.min()} to {s.max()}"
        if tag in PREFIX:
            vec = df[f"cvss_{tag}_vector"].dropna()
            bad = int((~vec.str.startswith(PREFIX[tag])).sum())
            line += f" | vectors not starting with {PREFIX[tag]}: {bad}"
        print(line)
        print("   severity:", df[f"cvss_{tag}_severity"].value_counts().to_dict())

    # 4. Rejected CVEs and rows with no score
    rejected = df["vuln_status"] == "Rejected"
    no_score = ~present.any(axis=1)
    print("\n4. Rejected CVEs")
    print(f"vuln_status == Rejected: {int(rejected.sum())} | of which is_kev=1: {int((rejected & kev).sum())}")
    print(f"Rows with no CVSS score: {int(no_score.sum())} | of which Rejected: {int((no_score & rejected).sum())}")
    print("No-score rows by vuln_status:")
    print(df.loc[no_score, "vuln_status"].value_counts().to_string())
    print("Sample descriptions of rejected CVEs:")
    for text in df.loc[rejected, "description"].head(3):
        print("  -", str(text)[:110])

    # 5. Columns that were not visible in the last report
    cols = [
        "description", "cwe_ids", "vendors", "products",
        "v3_integrity_impact", "v3_availability_impact",
        "v3_exploitability_score", "v3_impact_score",
    ]
    miss = pd.DataFrame(
        {
            "missing_pct": (df[cols].isna().mean() * 100).round(2),
            "missing_pct_kev": (df.loc[kev, cols].isna().mean() * 100).round(2),
        }
    )
    print("\n5. Missing % for the other columns")
    print(miss.to_string())
    print(f"\nSaved {REPORTS / 'cvss_version_patterns.csv'}")


if __name__ == "__main__":
    main()