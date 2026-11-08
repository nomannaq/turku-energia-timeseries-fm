"""Download the exact public checkpoint used in Experiment 2; no project data is uploaded."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "work_private/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, snapshot_download

MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


if __name__ == "__main__":
    destination = ROOT / "work_private/models/chronos-2"
    info = HfApi().model_info(MODEL_ID, revision=REVISION)
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=destination,
                      allow_patterns=["config.json", "model.safetensors", "README.md"])
    metadata = {"model_id": MODEL_ID, "revision": info.sha, "last_modified": str(info.last_modified),
                "downloaded_files": ["config.json", "model.safetensors", "README.md"]}
    (destination / "download_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(destination)
