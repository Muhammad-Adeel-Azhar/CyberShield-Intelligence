"""Collect CVE records from the NVD CVE API 2.0 and save every response untouched."""
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "nvd"
META_DIR = ROOT / "data" / "metadata"

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("NVD_API_KEY")

MAX_ATTEMPTS = 6
RETRY_STATUS = {403, 429, 500, 502, 503, 504}  # NVD answers 403/429 when rate limited


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "CyberShield-Intelligence/1.0 (university project)"
    session.headers["Accept"] = "application/json"
    if API_KEY:
        session.headers["apiKey"] = API_KEY
    return session


def fetch_page(session: requests.Session, start_index: int, results_per_page: int):
    """Fetch one page, retrying with growing waits on rate limits, server errors and network problems."""
    params = {"startIndex": start_index, "resultsPerPage": results_per_page}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait = min(10 * 2 ** (attempt - 1), 120)
        try:
            response = session.get(NVD_URL, params=params, timeout=(15, 180))
        except requests.exceptions.RequestException as exc:
            print(f"  network error ({type(exc).__name__}), retry {attempt}/{MAX_ATTEMPTS} in {wait}s")
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return response
        if response.status_code in RETRY_STATUS:
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, int(retry_after))
            print(f"  HTTP {response.status_code}, retry {attempt}/{MAX_ATTEMPTS} in {wait}s")
            time.sleep(wait)
            continue
        response.raise_for_status()  # other errors (400, 404, ...) are not retried

    raise RuntimeError(f"Giving up on startIndex={start_index} after {MAX_ATTEMPTS} attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect NVD CVE records")
    parser.add_argument("--resume", help="run id to continue, e.g. 20261118T101500Z")
    parser.add_argument("--max-pages", type=int, help="stop after this many pages (for testing)")
    parser.add_argument("--results-per-page", type=int, default=2000)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    run_id = args.resume or started.strftime("%Y%m%dT%H%M%SZ")
    run_dir = RAW_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    delay = 1.0 if API_KEY else 6.5
    print(f"Run {run_id} | API key: {'yes' if API_KEY else 'NO (slow mode)'} | pause {delay}s")

    session = make_session()
    total = None
    start = 0
    fetched = skipped = records = 0
    first_ts = last_ts = version = fmt = None

    while total is None or start < total:
        if args.max_pages and fetched + skipped >= args.max_pages:
            break

        path = run_dir / f"page_{start:07d}.json"
        just_fetched = False
        if path.exists():  # already downloaded in an earlier run
            content = path.read_bytes()
            skipped += 1
        else:
            content = fetch_page(session, start, args.results_per_page).content
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(content)
            tmp.replace(path)
            fetched += 1
            just_fetched = True

        data = json.loads(content)
        total = data["totalResults"]
        count = len(data["vulnerabilities"])
        records += count
        first_ts = first_ts or data.get("timestamp")
        last_ts = data.get("timestamp")
        version = data.get("version")
        fmt = data.get("format")

        page_no = start // args.results_per_page + 1
        pages = math.ceil(total / args.results_per_page)
        print(f"[{page_no}/{pages}] startIndex={start}  records={count}  total={total}")

        if count == 0:
            print("Empty page before reaching the total - stopping, check the run.")
            break
        start += args.results_per_page
        if just_fetched and start < total:
            time.sleep(delay)

    finished = datetime.now(timezone.utc)
    complete = total is not None and start >= total
    metadata = {
        "source": "NVD CVE API 2.0",
        "endpoint": NVD_URL,
        "run_id": run_id,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "api_key_used": bool(API_KEY),
        "results_per_page": args.results_per_page,
        "nvd_api_version": version,
        "nvd_format": fmt,
        "first_page_timestamp": first_ts,
        "last_page_timestamp": last_ts,
        "total_results_reported": total,
        "records_saved": records,
        "pages_downloaded_this_run": fetched,
        "pages_reused_from_earlier_run": skipped,
        "complete": complete,
        "raw_dir": run_dir.relative_to(ROOT).as_posix(),
    }
    meta_path = META_DIR / f"nvd_{run_id}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nMetadata saved: {meta_path}")
    print(f"Complete: {complete} | records {records} of {total}")
    if not complete:
        print(f"To continue later: python -m src.collect_nvd --resume {run_id}")


if __name__ == "__main__":
    main()