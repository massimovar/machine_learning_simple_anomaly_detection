import logging
import os
import sys
import threading
from datetime import datetime, timezone

import numpy as np
import yaml

import npu_evidence
from benchmark import run_benchmark
from detector import detect, load_threshold
from model import InferenceEngine, resolve_model_settings
from mqtt_client import create_client, publish_alert, publish_json, publish_status
from preprocessing import SampleBuffer, aggregate_window, load_scaler, normalise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("anomaly-detector")


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mqtt_cfg = config["mqtt"]
    prep_cfg = config["preprocessing"]
    model_cfg = config["model"]

    settings = resolve_model_settings(model_cfg)
    engine = InferenceEngine(settings["model_path"], use_npu=settings["use_npu"],
                             vx_delegate_path=settings["vx_delegate_path"])
    scaler = load_scaler(model_cfg["scaler_path"])
    threshold = load_threshold(settings["threshold_path"])
    buffer = SampleBuffer(prep_cfg["window_size"], prep_cfg["tag_names"])
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Benchmark state, shared by the MQTT thread and the benchmark thread:
    #   cpu_engine  - a second interpreter on the SAME model file, forced to CPU;
    #                 created on the first benchmark (costs RAM only if used).
    #   running     - True while a benchmark thread is alive; one at a time.
    #   last_vector - the last real, normalised window: benchmarking real data.
    # bench_lock protects "running" (check-and-set must be one atomic step).
    bench = {"cpu_engine": None, "running": False, "last_vector": None}
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
        window = buffer.add(sensor_data)
        if window is None:
            return

        feature_vector = aggregate_window(window)
        normalised = normalise(feature_vector, scaler)
        bench["last_vector"] = normalised
        result = detect(engine, normalised, threshold)

        message = f"score={result.score:.6f} threshold={result.threshold:.6f} on {engine.accelerator}"
        if result.is_anomaly:
            logger.warning("Anomaly detected: %s", message)
        else:
            logger.info("Normal: %s", message)

        evidence = {
            "accelerator": engine.accelerator,
            "precision": engine.precision,
            "inference_ms": round(engine.inference_ms, 4),
        }
        publish_alert(mqtt_client, mqtt_cfg["alert_topic"], result.to_alert_json(evidence))

    def benchmark_worker(runs: int) -> None:
        """Runs in its own thread: time CPU vs main engine, publish the result, free the slot."""
        try:
            if bench["cpu_engine"] is None:
                bench["cpu_engine"] = InferenceEngine(settings["model_path"], use_npu=False)
            vector = bench["last_vector"]
            if vector is None:
                # No window seen yet: the middle of the normal range.
                vector = np.full(model_cfg["input_size"], 0.5, dtype=np.float32)
            # Meanwhile the sensor handler keeps running: each reconstruct() takes
            # the engine lock, so a window arriving mid-benchmark waits < 1 ms.
            result = run_benchmark(engine, bench["cpu_engine"], vector, runs, npu_evidence.read_counters)
            result["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            logger.info("Benchmark: cpu p50 %.4f ms, npu %s", result["cpu"]["p50_ms"],
                        f"p50 {result['npu']['p50_ms']:.4f} ms" if result["npu"] else result["npu_unavailable"])
        except Exception as exc:
            logger.error("Benchmark failed: %s", exc)
            result = {"error": str(exc)}
        finally:
            # Always free the slot, even after an error, or no benchmark could run again.
            with bench_lock:
                bench["running"] = False
        publish_json(mqtt_client, mqtt_cfg["benchmark_topic"], result)
        # The first NPU compile may have just written a graph-cache file.
        publish_status(mqtt_client, mqtt_cfg["status_topic"], status_payload())

    def handle_command(command: dict) -> None:
        """Called by the MQTT client for every JSON message on the cmd topic."""
        if command.get("cmd") != "benchmark":
            logger.warning("Unknown command ignored: %s", command)
            return
        # Check-and-set under the lock: two quick clicks must not start two
        # benchmarks that would compete for the CPU and distort each other.
        with bench_lock:
            if bench["running"]:
                publish_json(mqtt_client, mqtt_cfg["benchmark_topic"], {"error": "benchmark already running"})
                return
            bench["running"] = True
        # WHY a thread: the MQTT loop must keep receiving the 1 Hz sensor data.
        # daemon=True: a running benchmark never blocks the container from stopping.
        # "runs" is clamped later by run_benchmark (1..MAX_RUNS).
        threading.Thread(target=benchmark_worker, args=(command.get("runs", 1000),), daemon=True).start()

    mqtt_client = create_client(
        broker_host=os.environ.get("MQTT_BROKER_HOST", mqtt_cfg["broker_host"]),
        broker_port=int(os.environ.get("MQTT_BROKER_PORT", mqtt_cfg["broker_port"])),
        client_id=mqtt_cfg["client_id"],
        sensor_topic=mqtt_cfg["sensor_topic"],
        on_sensor_message=handle_sensor_message,
        status_topic=mqtt_cfg["status_topic"],
        status_payload=status_payload,
        command_topic=mqtt_cfg["command_topic"],
        on_command=handle_command,
    )

    logger.info("Anomaly detector running")
    mqtt_client.loop_forever()


if __name__ == "__main__":
    main()
