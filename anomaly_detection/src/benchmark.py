"""
benchmark.py — the same model, the same input, CPU versus NPU
==============================================================

WHY THIS FILE EXISTS:
    A fair comparison changes ONE thing. Both engines load the same .tflite
    file and receive the same input vector; only the place where the math
    runs differs. Then:

    - different timings show the work really moved;
    - a small, non-zero output difference is expected: the NPU rounds
      intermediate int8 results in its own way, so it is a fingerprint of a
      different engine, not an error;
    - for this tiny model the CPU may well win: moving data to the NPU and
      back costs more than ~3,500 multiply-adds. That is the lesson, not a bug.

HOW A BENCHMARK CALL FLOWS:
    FTOptix button -> NpuStatusManager.RequestBenchmark() publishes
    {"cmd":"benchmark","runs":N} on anomaly/paperclip/cmd -> main.handle_command
    starts a worker thread -> run_benchmark() below -> the result is published
    on anomaly/paperclip/benchmark -> NpuStatusManager.ParseBenchmark() fills
    the Bench* variables on the panel.

WHAT ONE "RUN" MEASURES:
    One call to engine.reconstruct(vector) = one complete inference of batch 1,
    the same work the detector does for every sensor window:
        float -> int8 (quantize) -> set_tensor -> invoke() -> get_tensor
        -> int8 -> float (dequantize)
    On the NPU, invoke() also includes the trip CPU -> driver -> NPU -> CPU.
    The one-time graph compile (warm-up) happened when the engine was built,
    so it is NOT in these numbers.

WHY MORE RUNS DO NOT MAKE THE NPU WIN:
    Every run is an independent inference and pays the full trip again. Runs
    are repetitions of one measurement, not a bigger job: they make the
    percentiles more precise, they do not amortise anything. Measured on the
    device (p50 NPU / p50 CPU): 1,000 runs -> 2.20, 10,000 -> 2.19,
    50,000 -> 2.19. The NPU only wins when ONE invoke() carries more work
    (a bigger model or a batch of inputs). Measured with wider test models:
    the NPU beats this 1-thread CPU engine from about 2 M multiply-adds
    (ours: 3,328).
"""

import os
import time

import numpy as np

# Upper bound for one request. 5,000 runs give stable percentiles in ~6 s
# (both engines together); beyond that the ratio does not change, the panel
# just waits longer and one CPU core stays busy (it is shared with FTOptix).
# NpuStatusManager.RequestBenchmark() clamps to the same value.
MAX_RUNS = 5000


def time_runs(engine, vector: np.ndarray, runs: int) -> dict:
    """Run `runs` inferences of the same input on one engine and summarise the timings."""
    durations = []
    output = None
    for _ in range(runs):
        # perf_counter: the highest-resolution monotonic clock Python has.
        # The interval covers exactly one reconstruct() call, nothing else.
        start = time.perf_counter()
        output = engine.reconstruct(vector)
        durations.append((time.perf_counter() - start) * 1000.0)
    d = np.asarray(durations)
    # p50 (median) = the typical cost of one inference; a few slow runs
    # (the OS scheduling another process) cannot move it.
    # p95 = the "bad but normal" case: 95 % of the runs were faster.
    # mean is pulled up by those outliers; min = the best case with no noise.
    # The last output is kept (and removed before publishing) to compare engines.
    return {
        "accelerator": engine.accelerator,
        "runs": runs,
        "p50_ms": round(float(np.percentile(d, 50)), 4),
        "p95_ms": round(float(np.percentile(d, 95)), 4),
        "mean_ms": round(float(d.mean()), 4),
        "min_ms": round(float(d.min()), 4),
        "_output": output,
    }


def run_benchmark(main_engine, cpu_engine, vector: np.ndarray, runs: int) -> dict:
    """Time cpu_engine, then main_engine (the NPU one when available)."""
    # Never trust the request: at least 1 run, at most MAX_RUNS.
    runs = max(1, min(int(runs), MAX_RUNS))
    vector = np.asarray(vector, dtype=np.float32)

    # CPU first, then the main engine, one after the other (never in parallel,
    # so they do not compete for the CPU).
    cpu = time_runs(cpu_engine, vector, runs)
    main = time_runs(main_engine, vector, runs)

    # Same model, same input: the outputs must almost match. 0.0 on this model;
    # a large value would mean the NPU computes something else.
    difference = float(np.max(np.abs(cpu.pop("_output") - main.pop("_output"))))
    result = {
        "model": os.path.basename(main_engine.model_path),
        "precision": main_engine.precision,
        "runs": runs,
        "cpu": cpu,
        "npu": main if main_engine.accelerator == "npu" else None,
        "npu_unavailable": None if main_engine.accelerator == "npu" else (
            main_engine.npu_error or "NPU not requested (USE_NPU=false)"),
        "max_output_difference": round(difference, 6),
    }
    if result["npu"]:
        # > 1 means the NPU is slower than the CPU for one inference of this model.
        result["npu_vs_cpu_p50"] = round(main["p50_ms"] / cpu["p50_ms"], 2) if cpu["p50_ms"] else None
    return result
