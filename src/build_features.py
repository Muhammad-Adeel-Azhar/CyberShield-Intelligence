"""Day 2, Part C (step 1): build the modeling dataset and audit the planned features."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "nvd_kev_dataset.parquet"
OUT = ROOT / "data" / "processed" / "model_dataset.parquet"
META_DIR = ROOT / "data" / "metadata"

# Modeling population (see the Day 2 population checks)
FIRST_YEAR = 2016        # CVSS 3.x was assigned at publication from 2016 on
ONLY_ANALYZED = True     # False gives the robustness population (E)

NUMERIC = ["cvss_base_score", "published_year", "desc_words", "desc_chars"]
CATEGORICAL = ["attack_vector", "attack_complexity", "privileges_required", "user_interaction", "scope",
               "confidentiality_impact", "integrity_impact", "availability_impact", "cwe_category"]
V3_SOURCE = {name: f"v3_{name}" for name in CATEGORICAL[:-1]}
LOAD = ["cve_id", "published", "published_year", "vuln_status", "is_kev", "cvss_v31_score", "cvss_v30_score",
        "cwe_ids", "description"] + list(V3_SOURCE.values())


def rate_table(feat: pd.DataFrame, col: str, top: int | None = None) -> pd.DataFrame:
    t = feat.groupby(col, dropna=False).agg(rows=("is_kev", "size"), kev=("is_kev", "sum"))
    t["kev_pct"] = (t["kev"] / t["rows"] * 100).round(3)
    t = t.sort_values("rows", ascending=False)
    return t.head(top) if top else t


def main() -> None:
    df = pd.read_parquet(DATA, columns=LOAD)
    mask = (df["published_year"] >= FIRST_YEAR) & df["v3_attack_vector"].notna()
    if ONLY_ANALYZED:
        mask &= df["vuln_status"] == "Analyzed"
    pop = df[mask]

    feat = pd.DataFrame({"cve_id": pop["cve_id"], "published": pop["published"], "is_kev": pop["is_kev"]})
    feat["cvss_base_score"] = pop["cvss_v31_score"].combine_first(pop["cvss_v30_score"])
    feat["published_year"] = pop["published_year"].astype(int)
    feat["desc_words"] = pop["description"].str.split().str.len()
    feat["desc_chars"] = pop["description"].str.len()
    for name, source in V3_SOURCE.items():
        feat[name] = pop[source]
    feat["cwe_category"] = pop["cwe_ids"].str.split("|").str[0]  # first (primary) CWE
    feat = feat.sort_values("published").reset_index(drop=True)
    feat.to_parquet(OUT, index=False)

    kev = feat["is_kev"] == 1
    n, k = len(feat), int(kev.sum())
    pd.set_option("display.width", 200, "display.max_rows", None)
    label = "Published 2016+, CVSS 3.x data" + (", Analyzed" if ONLY_ANALYZED else "")
    print(f"1. Modeling population: {label}")
    print(f"Rows {n:,} | KEV {k:,} ({k / n * 100:.3f}%) | non-KEV {n - k:,} | ratio {(n - k) / k:.0f}:1")
    print("\nBy publication year (needed to choose the time-based split):")
    print(rate_table(feat, "published_year").sort_index().to_string())

    print("\n2. Missing values per feature (%), by class")
    miss = pd.DataFrame({
        "all": feat[NUMERIC + CATEGORICAL].isna().mean() * 100,
        "kev": feat.loc[kev, NUMERIC + CATEGORICAL].isna().mean() * 100,
        "non_kev": feat.loc[~kev, NUMERIC + CATEGORICAL].isna().mean() * 100,
    }).round(2)
    print(miss.to_string())

    print("\n3. Numerical features by class")
    print(feat.groupby("is_kev")[NUMERIC].agg(["mean", "median", "min", "max"]).round(2).T.to_string())

    print("\n4. Categorical features: rows, KEV and KEV rate per value")
    for col in CATEGORICAL[:-1]:
        print(f"\n{col}")
        print(rate_table(feat, col).to_string())

    print("\n5. CWE category (first CWE listed)")
    cwe = rate_table(feat, "cwe_category")
    print(f"Distinct values (incl. missing): {len(cwe)} | values with >= 50 rows: {int((cwe['rows'] >= 50).sum())}")
    covered = cwe[cwe["rows"] >= 50]["rows"].sum() / n * 100
    print(f"Rows covered by values with >= 50 rows: {covered:.1f}%")
    print("Top 12:")
    print(cwe.head(12).to_string())

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "population": label,
        "first_year": FIRST_YEAR,
        "only_analyzed": ONLY_ANALYZED,
        "rows": n,
        "kev": k,
        "kev_pct": round(k / n * 100, 4),
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "missing_pct": miss["all"].to_dict(),
        "output": OUT.relative_to(ROOT).as_posix(),
    }
    (META_DIR / "model_dataset_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()