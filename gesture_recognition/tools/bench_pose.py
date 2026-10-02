"""
bench_pose.py — measure MoveNet speed on this machine (CPU or NPU)
==================================================================

WHY THIS FILE EXISTS:
    Before the edge container is built we need real numbers: how fast is
    MoveNet on the i.MX 8M Plus CPU and NPU, and does the NPU keep the
    keypoints accurate?  This script loads the model exactly as the service
    does (src/pose.py), so the numbers match what the service will see.

    It prints:
        accelerator   the one actually used (if the NPU fails, it falls back to CPU)
        model load    includes the NPU graph compilation (can take seconds),
                      because src/pose.py tests the NPU with one inference
        first run     the first camera-sized frame after loading
        warm runs     median and 95th percentile of preprocess + inference
        keypoints     with --image: the 17 confidence scores.  Run once with
                      and once without --npu and compare; NPU scores near 0
                      mean the model's quantization does not suit the NPU.

USAGE (from the gesture_recognition folder):
    python tools/bench_pose.py                            # CPU, gray test frame
    python tools/bench_pose.py --npu                      # NPU (VX delegate)
    python tools/bench_pose.py --npu --image person.jpg   # also check keypoint scores
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pose import PoseEstimator  # noqa: E402

KEYPOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
                  "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
                  "l_knee", "r_knee", "l_ankle", "r_ankle"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoveNet the way the service runs it.")
    parser.add_argument("--npu", action="store_true", help="use the NXP VX delegate (falls back to CPU)")
    parser.add_argument("--vx-delegate", default="/usr/lib/libvx_delegate.so")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads (config.yaml num_threads)")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--image", help="photo of a person, to check the keypoint scores")
    parser.add_argument("--model", default=str(ROOT / "models" / "movenet_lightning_int8.tflite"))
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read image {args.image}")
    else:
        # WHY 640x480: the laptop stream size, so preprocessing costs the same.
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    t0 = time.perf_counter()
    estimator = PoseEstimator(args.model, use_npu=args.npu, vx_delegate_path=args.vx_delegate,
                              num_threads=args.threads)
    t1 = time.perf_counter()
    keypoints = estimator.infer(frame)
    t2 = time.perf_counter()

    times_ms = []
    for _ in range(args.runs):
        start = time.perf_counter()
        estimator.infer(frame)
        times_ms.append((time.perf_counter() - start) * 1000)

    print(f"accelerator  {estimator.accelerator} ({args.threads} CPU threads)")
    print(f"model load   {(t1 - t0) * 1000:.0f} ms")
    print(f"first run    {(t2 - t1) * 1000:.0f} ms")
    print(f"warm runs    p50 {statistics.median(times_ms):.1f} ms, "
          f"p95 {statistics.quantiles(times_ms, n=20)[18]:.1f} ms ({args.runs} runs)")
    if args.image:
        scores = keypoints[:, 2]
        print(f"keypoints    mean score {scores.mean():.2f}")
        print("             " + ", ".join(f"{n} {s:.2f}" for n, s in zip(KEYPOINT_NAMES, scores)))


if __name__ == "__main__":
    main()
