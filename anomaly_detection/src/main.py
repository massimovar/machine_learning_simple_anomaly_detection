import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone

import numpy as np
import yaml

import npu_evidence
from benchmark import run_benchmark
from model import InferenceEngine
from mqtt_client import create_client, publish
from preprocessing import SampleBuffer, load_scaler, normalise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("anomaly-detector")


def alert_payload(score: float, threshold: float, engine) -> dict:
    # The first three keys are the V1 contract; the evidence keys are appended after them.
    return {
        "is_anomaly": score > threshold,
        "anomaly_score": round(score, 6),
        "threshold": round(threshold, 6),
        "accelerator": engine.accelerator,
        "precision": engine.precision,
        "inference_ms": round(engine.inference_ms, 4),
    }


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    mqtt_cfg, prep_cfg, model_cfg = config["mqtt"], config["preprocessing"], config["model"]

    use_npu = os.environ.get("USE_NPU", str(model_cfg["use_npu"])).strip().lower() in ("1", "true", "yes", "on")
    # The NPU computes in integers, so it gets the int8 model; the CPU the float32 original.
    # If the NPU then fails, the SAME int8 model runs on the CPU, so the threshold still matches.
    variant = model_cfg["int8" if use_npu else "float32"]
    engine = InferenceEngine(variant["model_path"], use_npu=use_npu,
                             vx_delegate_path=os.environ.get("VX_DELEGATE_PATH", model_cfg["vx_delegate_path"]))
    # Benchmark reference: the SAME model file, forced to the CPU.
    cpu_engine = InferenceEngine(variant["model_path"], use_npu=False)
    with open(variant["threshold_path"], "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    scaler = load_scaler(model_cfg["scaler_path"])
    buffer = SampleBuffer(prep_cfg["window_size"], prep_cfg["tag_names"])
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # The benchmark times the last real window; before the first one, the middle of the normal range.
    last_vector = np.full(len(prep_cfg["tag_names"]), 0.5, dtype=np.float32)
    # Held while a benchmark runs: two quick clicks must not start two benchmarks
    # that would compete for the CPU and distort each other.
    bench_lock = threading.Lock()

    def status_payload() -> dict:
        return {
            "state": "online",
            "started_at": started_at,
            **engine.describe(),
            "threshold": round(threshold, 6),
            "galcore_version": npu_evidence.galcore_version(),
            "graph_cache": npu_evidence.graph_cache(),
        }

    def handle_sensor_message(sensor_data: dict) -> None:
        nonlocal last_vector
        window_mean = buffer.add(sensor_data)
        if window_mean is None:
            return
        last_vector = normalise(window_mean, scaler)
        alert = alert_payload(engine.score(last_vector), threshold, engine)
        message = f"score={alert['anomaly_score']:.6f} threshold={alert['threshold']:.6f} on {engine.accelerator}"
        if alert["is_anomaly"]:
            logger.warning("Anomaly detected: %s", message)
        else:
            logger.info("Normal: %s", message)
        publish(mqtt_client, mqtt_cfg["alert_topic"], alert)

    def benchmark_worker(runs: int) -> None:
        try:
            # Meanwhile the sensor handler keeps running: each reconstruct() takes
            # the engine lock, so a window arriving mid-benchmark waits < 1 ms.
            result = run_benchmark(engine, cpu_engine, last_vector, runs)
            result["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            logger.info("Benchmark: cpu p50 %.4f ms, npu %s", result["cpu"]["p50_ms"],
                        f"p50 {result['npu']['p50_ms']:.4f} ms" if result["npu"] else result["npu_unavailable"])
        except Exception as exc:
            logger.error("Benchmark failed: %s", exc)
            result = {"error": str(exc)}
        finally:
            bench_lock.release()
        publish(mqtt_client, mqtt_cfg["benchmark_topic"], result)
        # The first NPU compile may have just written a graph-cache file.
        publish(mqtt_client, mqtt_cfg["status_topic"], status_payload(), retain=True)

    def handle_command(command: dict) -> None:
        if command.get("cmd") != "benchmark":
            logger.warning("Unknown command ignored: %s", command)
            return
        if not bench_lock.acquire(blocking=False):
            publish(mqtt_client, mqtt_cfg["benchmark_topic"], {"error": "benchmark already running"})
            return
        # WHY a thread: the MQTT loop must keep receiving the 1 Hz sensor data.
        # daemon=True: a running benchmark never blocks the container from stopping.
        # "runs" is clamped by run_benchmark (1..MAX_RUNS).
        threading.Thread(target=benchmark_worker, args=(command.get("runs", 1000),), daemon=True).start()

    mqtt_client = create_client(
        broker_host=os.environ.get("MQTT_BROKER_HOST", mqtt_cfg["broker_host"]),
        broker_port=int(os.environ.get("MQTT_BROKER_PORT", mqtt_cfg["broker_port"])),
        client_id=mqtt_cfg["client_id"],
        sensor_topic=mqtt_cfg["sensor_topic"],
        on_sensor_message=handle_sensor_message,
        command_topic=mqtt_cfg["command_topic"],
        on_command=handle_command,
        status_topic=mqtt_cfg["status_topic"],
        status_payload=status_payload,
    )
    logger.info("Anomaly detector running")
    mqtt_client.loop_forever()


if __name__ == "__main__":
    main()
