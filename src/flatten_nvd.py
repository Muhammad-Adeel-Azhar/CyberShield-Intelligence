"""Flatten the raw NVD pages into one row per CVE and save as Parquet."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "metadata"
INTERIM_DIR = ROOT / "data" / "interim"

V3_FIELDS = {
    "attackVector": "attack_vector",
    "attackComplexity": "attack_complexity",
    "privilegesRequired": "privileges_required",
    "userInteraction": "user_interaction",
    "scope": "scope",
    "confidentialityImpact": "confidentiality_impact",
    "integrityImpact": "integrity_impact",
    "availabilityImpact": "availability_impact",
}


def latest_complete_run() -> tuple[str, Path]:
    for meta_path in sorted(META_DIR.glob("nvd_*.json"), reverse=True):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete"):
            return meta["run_id"], ROOT / meta["raw_dir"]
    raise SystemExit("No complete NVD run found in data/metadata")


def pick_metric(metrics: dict, key: str):
    """Return NVD's Primary metric for a CVSS version if present, otherwise the first one."""
    items = metrics.get(key) or []
    for item in items:
        if item.get("type") == "Primary":
            return item
    return items[0] if items else None


def flatten_cve(cve: dict, source_file: str) -> dict:
    metrics = cve.get("metrics") or {}
    description = next(
        (d.get("value") for d in cve.get("descriptions") or [] if d.get("lang") == "en"),
        None,
    )
    row = {
        "cve_id": cve.get("id"),
        "published": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "vuln_status": cve.get("vulnStatus"),
        "source_identifier": cve.get("sourceIdentifier"),
        "description": description,
    }

    # Score, severity and vector for every CVSS version that exists
    for key, tag in (
        ("cvssMetricV2", "v2"),
        ("cvssMetricV30", "v30"),
        ("cvssMetricV31", "v31"),
        ("cvssMetricV40", "v40"),
    ):
        metric = pick_metric(metrics, key) or {}
        data = metric.get("cvssData") or {}
        row[f"cvss_{tag}_score"] = data.get("baseScore")
        row[f"cvss_{tag}_severity"] = data.get("baseSeverity") or metric.get("baseSeverity")
        row[f"cvss_{tag}_vector"] = data.get("vectorString")

    # Detailed CVSS v3 components (3.1 preferred, otherwise 3.0)
    v3 = pick_metric(metrics, "cvssMetricV31") or pick_metric(metrics, "cvssMetricV30") or {}
    v3_data = v3.get("cvssData") or {}
    for src, dst in V3_FIELDS.items():
        row[f"v3_{dst}"] = v3_data.get(src)
    row["v3_exploitability_score"] = v3.get("exploitabilityScore")
    row["v3_impact_score"] = v3.get("impactScore")

    # Weaknesses (CWE)
    cwes = []
    for weakness in cve.get("weaknesses") or []:
        for d in weakness.get("description") or []:
            value = d.get("value")
            if d.get("lang") == "en" and value and value not in cwes:
                cwes.append(value)
    row["cwe_ids"] = "|".join(cwes) or None

    # Affected vendors and products from the CPE configuration
    vendors, products = set(), set()
    n_cpe = 0
    for config in cve.get("configurations") or []:
        for node in config.get("nodes") or []:
            for match in node.get("cpeMatch") or []:
                if not match.get("vulnerable"):
                    continue
                parts = (match.get("criteria") or "").split(":")
                if len(parts) > 4:
                    vendors.add(parts[3])
                    products.add(parts[4])
                    n_cpe += 1
    row["vendors"] = "|".join(sorted(vendors)) or None
    row["products"] = "|".join(sorted(products)) or None
    row["n_cpe_matches"] = n_cpe
    row["n_references"] = len(cve.get("references") or [])

    # For validating is_kev only - must NOT be used as a model feature (target leakage)
    row["nvd_cisa_exploit_add"] = cve.get("cisaExploitAdd")
    row["source_file"] = source_file
    return row


def main() -> None:
    run_id, raw_dir = latest_complete_run()
    pages = sorted(raw_dir.glob("page_*.json"))
    print(f"Run {run_id}: {len(pages)} pages in {raw_dir}")

    rows = []
    for page in tqdm(pages, unit="page"):
        data = json.loads(page.read_bytes())
        for item in data["vulnerabilities"]:
            rows.append(flatten_cve(item["cve"], page.name))

    df = pd.DataFrame(rows)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "nvd_flat.parquet"
    df.to_parquet(out, index=False)

    info = {
        "source_run_id": run_id,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        "unique_cve_ids": int(df["cve_id"].nunique()),
        "columns": list(df.columns),
        "output": out.relative_to(ROOT).as_posix(),
    }
    (META_DIR / f"flatten_{run_id}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nSaved {out}")
    print(f"Rows: {len(df)} | unique CVE ids: {info['unique_cve_ids']} | columns: {df.shape[1]}")


if __name__ == "__main__":
    main()