"""Download pinned research checkpoints; no project data is uploaded."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "work_private/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, snapshot_download

MODELS = [
    ("google/timesfm-3.0-pytorch", "43046b85ec22d584a13f8098c2ed39c889e129c2", "timesfm-3"),
    ("mistralai/Timestral-209M", "9874d252ab0aae16f7f7c999f98f98d6ad7d8365", "timestral-209m"),
]

if __name__ == "__main__":
    for model_id, revision, name in MODELS:
        destination = ROOT / "work_private/models" / name
        info = HfApi().model_info(model_id, revision=revision)
        snapshot_download(model_id, revision=revision, local_dir=destination,
                          allow_patterns=["*.json", "*.safetensors", "README.md", "LICENSE*"])
        metadata = {"model_id": model_id, "revision": info.sha, "last_modified": str(info.last_modified)}
        (destination / "download_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(destination)
