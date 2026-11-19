"""Day 2, Part C (step 0b): test the 2016+ era restriction and the vuln_status confound."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "nvd_kev_dataset.parquet"
COLS = ["cve_id", "published_year", "vuln_status", "is_kev", "v3_attack_vector"]


def row(name: str, mask: pd.Series, kev: pd.Series) -> dict:
    n, k = int(mask.sum()), int((mask & kev).sum())
    return {"population": name, "rows": n, "kev": k, "kev_pct": round(k / n * 100, 3),
            "ratio": f"{(n - k) / k:.0f}:1" if k else "n/a"}


def main() -> None:
    df = pd.read_parquet(DATA, columns=COLS)
    kev = df["is_kev"] == 1
    has_v3 = df["v3_attack_vector"].notna()
    analyzed = df["vuln_status"] == "Analyzed"
    era = df["published_year"] >= 2016
    pd.set_option("display.width", 200, "display.max_rows", None)

    print("1. Candidate populations")
    pops = [
        row("B. Has CVSS 3.x data (any year)", has_v3, kev),
        row("E. Published 2016+ and has CVSS 3.x", era & has_v3, kev),
        row("F. E and Analyzed", era & has_v3 & analyzed, kev),
    ]
    print(pd.DataFrame(pops).to_string(index=False))

    print("\n2. Older CVEs (before 2016) that have CVSS 3.x data")
    print(pd.DataFrame([row("Published before 2016, has CVSS 3.x", ~era & has_v3, kev)]).to_string(index=False))
    print(f"KEV CVEs published before 2016: {int((kev & ~era).sum())} of {int(kev.sum())}")

    print("\n3. Population E: vuln_status by class")
    e = df[era & has_v3]
    by_status = pd.crosstab(e["vuln_status"], e["is_kev"].map({0: "non_kev", 1: "kev"}))
    for col in ("kev", "non_kev"):
        if col not in by_status:
            by_status[col] = 0
    by_status["kev_rate_pct"] = (by_status["kev"] / (by_status["kev"] + by_status["non_kev"]) * 100).round(3)
    print(by_status.sort_values("non_kev", ascending=False).to_string())

    print("\n4. Population E: KEV rate by publication year")
    by_year = e.groupby("published_year").agg(cves=("cve_id", "size"), kev=("is_kev", "sum"))
    by_year["kev_pct"] = (by_year["kev"] / by_year["cves"] * 100).round(3)
    print(by_year.to_string())


if __name__ == "__main__":
    main()