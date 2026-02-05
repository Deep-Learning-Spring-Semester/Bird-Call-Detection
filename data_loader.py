"""
Data loader for bird call recordings from the Xeno-Canto API.
Fetches bird recordings from the Florida region and saves them locally.
Only includes species with at least 40 recordings.
"""

import os
import json
import time
import argparse
import requests
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict

load_dotenv()


API_BASE_URL = "https://xeno-canto.org/api/3/recordings"
# Florida bounding box (approximate): lat 24.5–31.0, lon -87.6–-80.0
FLORIDA_BOX = "24.5,-87.6,31.0,-80.0"
MIN_RECORDINGS_PER_SPECIES = 40
DATA_DIR = Path("data/recordings")
METADATA_DIR = Path("data/metadata")
PER_PAGE = 500  # max allowed by the API


def fetch_recordings(api_key: str, page: int = 1) -> dict:
    """Fetch a page of bird recordings from Florida via the Xeno-Canto API."""
    params = {
        "query": f"box:{FLORIDA_BOX} grp:birds",
        "key": api_key,
        "per_page": PER_PAGE,
        "page": page,
    }
    resp = requests.get(API_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_recordings(api_key: str) -> list[dict]:
    """Paginate through the API and collect all Florida bird recordings."""
    first_page = fetch_recordings(api_key, page=1)
    num_pages = first_page["numPages"]
    total = first_page["numRecordings"]
    print(f"Found {total} recordings across {num_pages} page(s)")

    all_recordings = list(first_page["recordings"])

    for page in range(2, num_pages + 1):
        print(f"  Fetching page {page}/{num_pages}...")
        data = fetch_recordings(api_key, page=page)
        all_recordings.extend(data["recordings"])
        time.sleep(1)  # be polite to the server

    return all_recordings


def group_by_species(recordings: list[dict]) -> dict[str, list[dict]]:
    """Group recordings by species english name."""
    species_map = defaultdict(list)
    for rec in recordings:
        name = rec.get("en", "Unknown")
        species_map[name].append(rec)
    return dict(species_map)


def filter_species(species_map: dict[str, list[dict]], min_count: int) -> dict[str, list[dict]]:
    """Keep only species that have at least `min_count` recordings."""
    filtered = {
        name: recs
        for name, recs in species_map.items()
        if len(recs) >= min_count
    }
    excluded = len(species_map) - len(filtered)
    print(f"Kept {len(filtered)} species (>= {min_count} recordings), excluded {excluded}")
    return filtered


def download_recording(rec: dict, species_dir: Path) -> bool:
    """Download a single recording audio file. Returns True on success."""
    file_url = rec.get("file")
    if not file_url:
        return False

    # Ensure the URL has a scheme
    if file_url.startswith("//"):
        file_url = "https:" + file_url

    rec_id = rec["id"]
    file_name = rec.get("file-name", f"{rec_id}.mp3")
    dest = species_dir / file_name

    if dest.exists():
        return True

    try:
        resp = requests.get(file_url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"    Failed to download recording {rec_id}: {e}")
        return False


def save_metadata(species_map: dict[str, list[dict]]):
    """Save per-species metadata JSON files."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, recs in species_map.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        path = METADATA_DIR / f"{safe_name}.json"
        with open(path, "w") as f:
            json.dump({"species": name, "count": len(recs), "recordings": recs}, f, indent=2)
    print(f"Saved metadata for {len(species_map)} species to {METADATA_DIR}/")


def download_all(species_map: dict[str, list[dict]], max_per_species: int | None = None):
    """Download audio files for every species, organized into subdirectories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total_species = len(species_map)

    for idx, (name, recs) in enumerate(sorted(species_map.items()), 1):
        safe_name = name.replace(" ", "_").replace("/", "_")
        species_dir = DATA_DIR / safe_name
        species_dir.mkdir(parents=True, exist_ok=True)

        subset = recs[:max_per_species] if max_per_species else recs
        print(f"[{idx}/{total_species}] {name}: downloading {len(subset)} recordings...")

        success = 0
        for rec in subset:
            if download_recording(rec, species_dir):
                success += 1
            time.sleep(0.5)  # rate-limit downloads

        print(f"    Downloaded {success}/{len(subset)}")


def main():
    parser = argparse.ArgumentParser(description="Download Florida bird call recordings from Xeno-Canto")
    parser.add_argument("--api-key", default=os.environ.get("XENO_CANTO_API_KEY"),
                        help="Xeno-Canto API key (or set XENO_CANTO_API_KEY env var)")
    parser.add_argument("--min-recordings", type=int, default=MIN_RECORDINGS_PER_SPECIES,
                        help=f"Minimum recordings per species (default: {MIN_RECORDINGS_PER_SPECIES})")
    parser.add_argument("--max-per-species", type=int, default=None,
                        help="Cap downloads per species (default: no cap)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only fetch and save metadata, skip audio downloads")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("API key required. Pass --api-key or set XENO_CANTO_API_KEY env var. "
                      "Register at https://xeno-canto.org to obtain one.")

    print("Fetching Florida bird recordings from Xeno-Canto...")
    all_recs = fetch_all_recordings(args.api_key)

    species_map = group_by_species(all_recs)
    print(f"Total species found: {len(species_map)}")

    filtered = filter_species(species_map, args.min_recordings)
    save_metadata(filtered)

    if not args.metadata_only:
        download_all(filtered, args.max_per_species)
        print("Done.")
    else:
        print("Metadata saved. Skipping audio downloads (--metadata-only).")


if __name__ == "__main__":
    main()
