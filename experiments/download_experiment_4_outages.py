"""Download public Nord Pool UMM notice histories, retaining every revision."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work_private/experiment4"
RAW = OUT / "outage_raw"
BASE = "https://ummapi.nordpoolgroup.com/messages"


def get(url, path, params=None):
    if path.exists():
        return json.loads(path.read_text())
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    data = response.json()
    path.write_text(json.dumps(data))
    time.sleep(0.1)
    return data


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    common = {"areas": "10YFI-1--------U", "includeOutdated": "true"}
    searches = [
        {**common, "eventStartDate": "2026-05-31T00:00:00Z", "eventStopDate": "2026-10-05T12:00:00Z"},
        {**common, "publicationStartDate": "2026-05-01T00:00:00Z", "publicationStopDate": "2026-10-04T21:00:00Z"},
    ]
    ids = set()
    for i, params in enumerate(searches):
        data = get(BASE, RAW / f"search_{i}.json", params)
        assert data["total"] == len(data["items"]), "Incomplete outage listing"
        ids.update(m["messageId"] for m in data["items"] if m["messageType"] in (1, 2, 3))
    print(f"Fetching complete revision histories for {len(ids)} notices", flush=True)
    def history(message_id):
        data = get(f"{BASE}/{message_id}", RAW / f"{message_id}.json")
        assert isinstance(data, list)
        assert all(m["messageId"] == message_id for m in data)
        return data
    messages = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, revisions in enumerate(pool.map(history, sorted(ids)), 1):
            messages.extend(revisions)
            if n % 50 == 0 or n == len(ids):
                print(f"Outage histories {n}/{len(ids)}, revisions {len(messages)}", flush=True)
    keys = [(m["messageId"], m["version"]) for m in messages]
    assert len(keys) == len(set(keys))
    (OUT / "outage_revisions.json").write_text(json.dumps(messages))
    manifest = {"endpoint": BASE, "searches": searches, "notices": len(ids), "revisions": len(messages),
                "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.glob("*.json"))}}
    (OUT / "outage_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
