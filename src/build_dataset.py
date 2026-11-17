"""Match NVD records against the CISA KEV catalog and create the is_kev label."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_KEV = ROOT / "data" / "raw" / "kev"
INTERIM = ROOT / "data" / "interim"
META_DIR = ROOT / "data" / "metadata"


def load_kev() -> tuple[Path, dict, pd.DataFrame]:
    path = sorted(RAW_KEV.glob("kev_*.json"))[-1]  # newest download
    catalog = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "cve_id": (v.get("cveID") or v.get("cveId") or "").strip().upper(),
            "kev_date_added": v.get("dateAdded"),
        }
        for v in catalog["vulnerabilities"]
    ]
    return path, catalog, pd.DataFrame(rows)


def main() -> None:
    nvd = pd.read_parquet(INTERIM / "nvd_flat.parquet")
    nvd["cve_id"] = nvd["cve_id"].str.strip().str.upper()

    kev_path, catalog, kev = load_kev()
    kev_rows = len(kev)
    kev = kev.drop_duplicates("cve_id")

    # Match on CVE ID (many NVD rows -> at most one KEV row, so row count must not change)
    merged = nvd.merge(kev, on="cve_id", how="left")
    assert len(merged) == len(nvd), "merge changed the number of rows"
    merged["is_kev"] = merged["cve_id"].isin(kev["cve_id"]).astype(int)

    # Checks
    unmatched = sorted(set(kev["cve_id"]) - set(nvd["cve_id"]))
    kev_matched = int(merged.loc[merged["is_kev"] == 1, "cve_id"].nunique())
    nvd_flag = merged["nvd_cisa_exploit_add"].notna()
    both = int(((merged["is_kev"] == 1) & nvd_flag).sum())
    only_ours = int(((merged["is_kev"] == 1) & ~nvd_flag).sum())
    only_nvd = int(((merged["is_kev"] == 0) & nvd_flag).sum())

    out = INTERIM / "nvd_kev_merged.parquet"
    merged.to_parquet(out, index=False)

    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "kev_file": kev_path.name,
        "kev_catalog_version": catalog.get("catalogVersion"),
        "kev_catalog_date_released": catalog.get("dateReleased"),
        "kev_rows_in_file": kev_rows,
        "kev_unique_cve_ids": len(kev),
        "nvd_rows": len(nvd),
        "nvd_unique_cve_ids": int(nvd["cve_id"].nunique()),
        "rows_flagged_is_kev": int(merged["is_kev"].sum()),
        "kev_cves_matched_in_nvd": kev_matched,
        "kev_cves_not_found_in_nvd": unmatched,
        "crosscheck_both": both,
        "crosscheck_only_in_our_match": only_ours,
        "crosscheck_only_in_nvd_flag": only_nvd,
        "output": out.relative_to(ROOT).as_posix(),
    }
    (META_DIR / "merge_kev.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"NVD rows: {len(nvd)} | unique CVE ids: {info['nvd_unique_cve_ids']}")
    print(f"KEV file: {kev_path.name} (version {info['kev_catalog_version']}), {len(kev)} unique CVEs")
    print(f"Rows flagged is_kev=1: {info['rows_flagged_is_kev']} | KEV CVEs matched in NVD: {kev_matched}")
    print(f"KEV CVEs not found in NVD: {len(unmatched)} {unmatched[:20]}")
    print(f"Cross-check vs NVD cisaExploitAdd -> both: {both}, only ours: {only_ours}, only NVD: {only_nvd}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()