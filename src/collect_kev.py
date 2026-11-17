"""Download the CISA KEV catalog and record collection metadata."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "kev"
META_DIR = ROOT / "data" / "metadata"


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "CyberShield-Intelligence/1.0 (university project)"
    return session


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    collected_at = datetime.now(timezone.utc)
    stamp = collected_at.strftime("%Y%m%dT%H%M%SZ")

    response = make_session().get(KEV_URL, timeout=60)
    response.raise_for_status()

    # Save the original bytes untouched
    raw_path = RAW_DIR / f"kev_{stamp}.json"
    raw_path.write_bytes(response.content)

    catalog = response.json()
    records = len(catalog["vulnerabilities"])
    if catalog["count"] != records:
        print(f"WARNING: catalog says {catalog['count']} but found {records} records")

    metadata = {
        "source": "CISA Known Exploited Vulnerabilities catalog",
        "url": KEV_URL,
        "collected_at_utc": collected_at.isoformat(),
        "http_status": response.status_code,
        "catalog_title": catalog.get("title"),
        "catalog_version": catalog.get("catalogVersion"),
        "date_released": catalog.get("dateReleased"),
        "reported_count": catalog["count"],
        "records_found": records,
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "raw_file": str(raw_path.relative_to(ROOT)),
    }
    meta_path = META_DIR / f"kev_{stamp}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved raw file : {raw_path}")
    print(f"Saved metadata : {meta_path}")
    print(f"Catalog version: {metadata['catalog_version']}  ({records} CVEs)")


if __name__ == "__main__":
    main()