"""
npu_check.py — one-shot proof that this container can (or cannot) use the NPU
==============================================================================

WHY THIS FILE EXISTS:
    Before starting the full detector (MQTT, broker, FTOptix), check one thing
    only: can THIS image load the VX delegate and run OUR model on the NPU?
    It uses exactly the production code (InferenceEngine + run_benchmark),
    so a pass here means the detector will find the NPU too.

    It needs no broker and writes nothing outside the container (the graph
    cache only if VIV_VX_CACHE_BINARY_GRAPH_DIR points to a mounted volume).

RUN (on the device, image already loaded):
    docker run --rm --device /dev/galcore -v /usr/lib:/host/usr/lib:ro \
        localhost/anomaly_detection-detector:npu python tools/npu_check.py

    Options: --model PATH  --runs N  --cpu-only (negative control)

READ THE RESULT:
    "accelerator": "npu" and an empty "npu_error"  -> the NPU ran our model
    "delegation"   -> how many graph nodes the VX delegate took over
    "warmup_ms"    -> first run; on the NPU it includes compiling the graph
    "benchmark"    -> same model, same input: CPU vs NPU, plus the largest
                      output difference between the two (should be tiny)
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402

import npu_evidence  # noqa: E402
from benchmark import run_benchmark  # noqa: E402
from model import DEFAULT_VX_DELEGATE, InferenceEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=os.path.join(HERE, "..", "models", "autoencoder_int8.tflite"))
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--cpu-only", action="store_true", help="negative control: do not request the NPU")
    args = parser.parse_args()

    delegate = os.environ.get("VX_DELEGATE_PATH", DEFAULT_VX_DELEGATE)
    engine = InferenceEngine(args.model, use_npu=not args.cpu_only, vx_delegate_path=delegate)
    cpu_engine = InferenceEngine(args.model, use_npu=False)
    vector = np.full(engine.interpreter.get_input_details()[0]["shape"][-1], 0.5, dtype=np.float32)

    report = {
        "engine": engine.describe(),
        "galcore_version": npu_evidence.galcore_version(),
        "dev_galcore": os.path.exists("/dev/galcore"),
        "benchmark": run_benchmark(engine, cpu_engine, vector, args.runs),
        "graph_cache": npu_evidence.graph_cache(),
    }
    print(json.dumps(report, indent=2))
    print(f"\nRESULT: inference ran on the {engine.accelerator.upper()}"
          + (f"  (NPU error: {engine.npu_error})" if engine.npu_error else ""))
    return 0 if (engine.accelerator == "npu") != args.cpu_only else 1


if __name__ == "__main__":
    sys.exit(main())
