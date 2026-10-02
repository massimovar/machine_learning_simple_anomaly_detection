"""
fetch_model.py — download the MoveNet pose model and verify its checksum
=========================================================================

WHY THIS FILE EXISTS:
    The model (2.9 MB) is not committed to Git. This script downloads the
    exact file used by NXP's eIQ Model Zoo (MoveNet SinglePose Lightning,
    int8, version 4) and checks its SHA-256, so the laptop, the container
    and the edge device all run byte-identical weights.

    Model facts (checked with the TFLite interpreter):
        input   [1, 192, 192, 3]  uint8    RGB image
        output  [1, 1, 17, 3]     float32  17 body keypoints x (y, x, score)

USAGE:
    python tools/fetch_model.py                   # -> models/movenet_lightning_int8.tflite
    python tools/fetch_model.py path/to/x.tflite  # custom destination

LICENSE:
    MoveNet is published by Google under the Apache 2.0 license.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4"
    "?lite-format=tflite"
)
MODEL_SHA256 = "cd7cc22fa946e5d146a7b98d496853e1923e22828d3972d579973f27f91bb105"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models" / "movenet_lightning_int8.tflite"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(target: Path) -> None:
    if target.exists() and sha256_of(target.read_bytes()) == MODEL_SHA256:
        print(f"Model already present: {target}")
        return

    print(f"Downloading MoveNet from {MODEL_URL} ...")
    # WHY a User-Agent: the URL redirects to Kaggle storage, which may
    # reject requests that do not look like a normal client.
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()

    digest = sha256_of(data)
    if digest != MODEL_SHA256:
        sys.exit(f"Checksum mismatch: expected {MODEL_SHA256}, got {digest}. Nothing saved.")

    target.parent.mkdir(parents=True, exist_ok=True)
    # WHY write-then-rename: an interrupted download never leaves a broken
    # file with the model's name behind.
    partial = target.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(target)
    print(f"Saved {len(data)} bytes to {target}")


if __name__ == "__main__":
    fetch(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH)
