"""Save the final cleaned dataset (rejected CVEs removed) and a summary for the team."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
META_DIR = ROOT / "data" / "metadata"

# Columns that must never be model inputs
NOT_FEATURES = ["cve_id", "source_file", "kev_date_added", "nvd_cisa_exploit_add"]
# Columns that may carry the label indirectly
SUSPECT_FEATURES = ["last_modified", "vuln_status"]


def newest_json(pattern: str, must_be_complete: bool = False) -> dict:
    for path in sorted(META_DIR.glob(pattern), reverse=True):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if not must_be_complete or meta.get("complete"):
            return meta
    raise SystemExit(f"No metadata file matching {pattern}")


def main() -> None:
    df = pd.read_parquet(INTERIM / "nvd_kev_clean.parquet")
    rows_in = len(df)
    kev_in = int(df["is_kev"].sum())

    rejected = df["vuln_status"] == "Rejected"
    assert int((rejected & (df["is_kev"] == 1)).sum()) == 0, "a KEV CVE is marked Rejected"
    df = df[~rejected].sort_values("published").reset_index(drop=True)

    # Final checks
    assert df["cve_id"].notna().all() and df["cve_id"].is_unique, "CVE ids are not unique"
    assert set(df["is_kev"].unique()) <= {0, 1}, "is_kev has unexpected values"
    assert int(df["is_kev"].sum()) == kev_in, "KEV rows were lost"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "nvd_kev_dataset.parquet"
    df.to_parquet(out, index=False)
    sha256 = hashlib.sha256(out.read_bytes()).hexdigest()

    nvd_meta = newest_json("nvd_*.json", must_be_complete=True)
    kev_meta = newest_json("kev_*.json")
    kev_count = int(df["is_kev"].sum())

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": out.relative_to(ROOT).as_posix(),
        "file_size_mb": round(out.stat().st_size / 1e6, 1),
        "sha256": sha256,
        "rows_before_filtering": rows_in,
        "rejected_removed": rows_in - len(df),
        "rows": len(df),
        "columns": df.shape[1],
        "kev_count": kev_count,
        "non_kev_count": len(df) - kev_count,
        "kev_pct": round(kev_count / len(df) * 100, 4),
        "published_min": str(df["published"].min()),
        "published_max": str(df["published"].max()),
        "sources": {
            "nvd_run_id": nvd_meta["run_id"],
            "nvd_last_page_timestamp": nvd_meta["last_page_timestamp"],
            "kev_catalog_version": kev_meta["catalog_version"],
            "kev_date_released": kev_meta["date_released"],
        },
        "do_not_use_as_features": NOT_FEATURES,
        "suspect_features": SUSPECT_FEATURES,
        "notes": [
            "All KEV CVEs have vuln_status == Analyzed, so vuln_status leaks the label and must not be a feature.",
            "is_kev is a snapshot label: recent CVEs labeled 0 may be added to KEV later.",
            "Use a time-based split with a cutoff that leaves labels time to mature.",
            "Rejected CVEs (vuln_status == Rejected) were removed.",
        ],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    (META_DIR / "dataset_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Rows: {rows_in} -> {len(df)} (removed {rows_in - len(df)} rejected)")
    print(f"is_kev=1: {kev_count} ({info['kev_pct']}%) | is_kev=0: {info['non_kev_count']}")
    print(f"Columns: {df.shape[1]} | published {info['published_min'][:10]} to {info['published_max'][:10]}")
    print(f"Saved {out} ({info['file_size_mb']} MB)")
    print(f"SHA-256: {sha256}")


if __name__ == "__main__":
    main()