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
"""

import time

import numpy as np

MAX_RUNS = 5000


def time_runs(engine, vector: np.ndarray, runs: int) -> dict:
    durations = []
    output = None
    for _ in range(runs):
        start = time.perf_counter()
        output = engine.reconstruct(vector)
        durations.append((time.perf_counter() - start) * 1000.0)
    d = np.asarray(durations)
    return {
        "accelerator": engine.accelerator,
        "runs": runs,
        "p50_ms": round(float(np.percentile(d, 50)), 4),
        "p95_ms": round(float(np.percentile(d, 95)), 4),
        "mean_ms": round(float(d.mean()), 4),
        "min_ms": round(float(d.min()), 4),
        "_output": output,
    }


def run_benchmark(main_engine, cpu_engine, vector: np.ndarray, runs: int, read_counters=None) -> dict:
    """Time cpu_engine, then main_engine (the NPU one when available).

    read_counters: optional callable returning the driver counters; it is
    called right before and right after the main engine's runs.
    """
    runs = max(1, min(int(runs), MAX_RUNS))
    vector = np.asarray(vector, dtype=np.float32)

    cpu = time_runs(cpu_engine, vector, runs)
    before = read_counters() if read_counters else None
    main = time_runs(main_engine, vector, runs)
    after = read_counters() if read_counters else None

    difference = float(np.max(np.abs(cpu.pop("_output") - main.pop("_output"))))
    result = {
        "model": main_engine.describe()["model"],
        "precision": main_engine.precision,
        "runs": runs,
        "cpu": cpu,
        "npu": main if main_engine.accelerator == "npu" else None,
        "npu_unavailable": None if main_engine.accelerator == "npu" else (
            main_engine.npu_error or "NPU not requested (USE_NPU=false)"),
        "max_output_difference": round(difference, 6),
        "counters_before": before,
        "counters_after": after,
    }
    if result["npu"]:
        result["npu_vs_cpu_p50"] = round(main["p50_ms"] / cpu["p50_ms"], 2) if cpu["p50_ms"] else None
    return result
