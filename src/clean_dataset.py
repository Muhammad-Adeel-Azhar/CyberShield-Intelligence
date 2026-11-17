"""Remove duplicate CVE records and convert date columns."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
META_DIR = ROOT / "data" / "metadata"

DATE_COLUMNS = ["published", "last_modified", "kev_date_added", "nvd_cisa_exploit_add"]


def main() -> None:
    df = pd.read_parquet(INTERIM / "nvd_kev_merged.parquet")
    rows_start = len(df)
    kev_start = int(df["is_kev"].sum())

    # ---- 1. CVE ID format and duplicates ----
    df["cve_id"] = df["cve_id"].str.strip().str.upper()
    invalid_ids = int((~df["cve_id"].str.match(r"^CVE-\d{4}-\d{4,}$", na=False)).sum())
    full_row_duplicates = int(df.duplicated().sum())
    duplicate_ids = int(df.duplicated("cve_id").sum())

    # If an ID appears more than once, keep the most recently modified record
    df = df.sort_values("last_modified", ascending=False).drop_duplicates("cve_id", keep="first")

    # ---- 2. Convert dates (unparseable values become NaT and are counted) ----
    date_report = {}
    for col in DATE_COLUMNS:
        before = int(df[col].notna().sum())
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        after = int(df[col].notna().sum())
        date_report[col] = {
            "non_null": after,
            "failed_to_parse": before - after,
            "min": str(df[col].min()),
            "max": str(df[col].max()),
        }

    df["published_year"] = df["published"].dt.year.astype("Int64")
    df = df.sort_values("published").reset_index(drop=True)

    modified_before_published = int((df["last_modified"] < df["published"]).sum())
    published_in_future = int((df["published"] > pd.Timestamp.now(tz="UTC")).sum())

    out = INTERIM / "nvd_kev_clean.parquet"
    df.to_parquet(out, index=False)

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_before": rows_start,
        "invalid_cve_ids": invalid_ids,
        "full_row_duplicates": full_row_duplicates,
        "duplicate_cve_ids_removed": duplicate_ids,
        "rows_after": len(df),
        "is_kev_before": kev_start,
        "is_kev_after": int(df["is_kev"].sum()),
        "dates": date_report,
        "last_modified_before_published": modified_before_published,
        "published_in_future": published_in_future,
        "output": out.relative_to(ROOT).as_posix(),
    }
    (META_DIR / "clean_dedup_dates.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Rows: {rows_start} -> {len(df)} | invalid CVE ids: {invalid_ids}")
    print(f"Full-row duplicates: {full_row_duplicates} | duplicate CVE ids removed: {duplicate_ids}")
    print(f"is_kev=1: {kev_start} -> {info['is_kev_after']}")
    for col, rep in date_report.items():
        print(f"{col}: non-null {rep['non_null']}, failed to parse {rep['failed_to_parse']}, "
              f"range {rep['min'][:10]} to {rep['max'][:10]}")
    print(f"last_modified earlier than published: {modified_before_published} | published in the future: {published_in_future}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()