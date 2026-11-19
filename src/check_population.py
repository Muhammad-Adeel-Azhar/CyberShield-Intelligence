"""Day 2, Part C (step 0): compare candidate modeling populations and check the CVSS 3.x shortcut."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "nvd_kev_dataset.parquet"
COLS = ["cve_id", "published_year", "vuln_status", "is_kev", "v3_attack_vector"]


def main() -> None:
    df = pd.read_parquet(DATA, columns=COLS)
    kev = df["is_kev"] == 1
    has_v3 = df["v3_attack_vector"].notna()
    analyzed = df["vuln_status"] == "Analyzed"
    pd.set_option("display.width", 200, "display.max_rows", None)

    pops = {
        "A. All CVEs": pd.Series(True, index=df.index),
        "B. Has CVSS 3.x data": has_v3,
        "C. Analyzed only": analyzed,
        "D. Analyzed and has CVSS 3.x": analyzed & has_v3,
    }
    rows = []
    for name, mask in pops.items():
        n, k = int(mask.sum()), int((mask & kev).sum())
        rows.append({"population": name, "rows": n, "kev": k, "kev_pct": round(k / n * 100, 3),
                     "ratio": f"{(n - k) / k:.0f}:1"})
    print("1. Candidate modeling populations")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n2. The CVSS 3.x shortcut")
    print(f"KEV CVEs with CVSS 3.x data    : {int((kev & has_v3).sum()):,} of {int(kev.sum()):,}")
    print(f"KEV CVEs WITHOUT CVSS 3.x data : {int((kev & ~has_v3).sum()):,}")
    print(f"Non-KEV CVEs without CVSS 3.x  : {int((~kev & ~has_v3).sum()):,} ({(~kev & ~has_v3).sum() / (~kev).sum() * 100:.1f}% of non-KEV)")

    print("\n3. Attack vector, only CVEs that have CVSS 3.x data (% within each class)")
    share = pd.crosstab(df.loc[has_v3, "v3_attack_vector"], df.loc[has_v3, "is_kev"].map({0: "non-KEV", 1: "KEV"}),
                        normalize="columns") * 100
    print(share.round(1).to_string())

    print("\n4. CVSS 3.x availability by publication year")
    by_year = df.groupby("published_year").agg(
        cves=("cve_id", "size"), pct_with_v3=("v3_attack_vector", lambda s: round(s.notna().mean() * 100, 1)))
    by_year["kev_with_v3"] = df[kev & has_v3].groupby("published_year").size()
    by_year["kev_without_v3"] = df[kev & ~has_v3].groupby("published_year").size()
    print(by_year.fillna(0).astype({"kev_with_v3": int, "kev_without_v3": int}).tail(18).to_string())


if __name__ == "__main__":
    main()