"""Print the key facts of the final dataset so every team member reviews the same numbers."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "nvd_kev_dataset.parquet"
SUMMARY = ROOT / "data" / "metadata" / "dataset_summary.json"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"Dataset not found: {DATA}\nDownload it into data\\processed first.")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    print("Checksum matches dataset_summary.json:", sha == summary["sha256"])

    df = pd.read_parquet(DATA)
    kev = df["is_kev"] == 1
    pd.set_option("display.width", 200, "display.max_columns", None)

    print(f"\nRows: {len(df)} | columns: {df.shape[1]}")
    print(f"Unique CVE ids: {df['cve_id'].nunique()} | duplicate rows: {int(df.duplicated().sum())}")
    print(f"is_kev=1: {int(kev.sum())} ({kev.mean() * 100:.3f}%) | is_kev=0: {int((~kev).sum())}")
    print(f"Published: {df['published'].min():%Y-%m-%d} to {df['published'].max():%Y-%m-%d}")
    print(f"Rejected rows left: {int((df['vuln_status'] == 'Rejected').sum())}")
    print(f"vuln_status of KEV rows: {df.loc[kev, 'vuln_status'].value_counts().to_dict()}")

    has_v31 = df["cvss_v31_score"].notna()
    print(f"\nCVSS 3.1 score present: all {has_v31.mean() * 100:.1f}% | "
          f"KEV {has_v31[kev].mean() * 100:.1f}% | non-KEV {has_v31[~kev].mean() * 100:.1f}%")

    miss = df.drop(columns=["kev_date_added", "nvd_cisa_exploit_add"]).isna().mean().mul(100).round(2)
    print("\nColumns with the most missing values (%):")
    print(miss.sort_values(ascending=False).head(8).to_string())

    cols = ["cve_id", "published", "vuln_status", "cvss_v31_score", "cwe_ids", "is_kev"]
    print("\n5 random KEV rows (spot-check them on nvd.nist.gov):")
    print(df.loc[kev, cols].sample(5, random_state=1).to_string(index=False))
    print("\n5 random non-KEV rows:")
    print(df.loc[~kev, cols].sample(5, random_state=1).to_string(index=False))

    print("\nDo NOT use as features:", summary["do_not_use_as_features"])
    print("Suspect (possible label leak):", summary["suspect_features"])
    print("\nNotes:")
    for note in summary["notes"]:
        print(" -", note)


if __name__ == "__main__":
    main()