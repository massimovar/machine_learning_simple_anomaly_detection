"""
main.py — Entry point for the anomaly detection service
========================================================

WHY THIS FILE EXISTS:
    This is the "glue" that wires everything together.  It:
    1. Loads configuration from config.yaml
    2. Loads the trained model, scaler, and threshold from disk
    3. Connects to the MQTT broker and subscribes to sensor data
    4. For every incoming message: buffers → windows → preprocesses → detects → alerts

    Think of it as the conductor of an orchestra: it doesn't play any instrument
    itself, but it tells each module when and what to play.

HOW THE REAL-TIME LOOP WORKS:
    Messages arrive one per second.  Each message contains 10 sensor values.
    We collect them in a buffer (SampleBuffer).  Every 30 messages, the buffer
    emits a "window".  We average the window into a single 10-value vector,
    normalise it, feed it to the autoencoder, and check the reconstruction
    error against our threshold.  If it's anomalous, we publish an alert.

    Timeline (at 1 Hz):
    second 1-30   → buffering ...
    second 30     → window ready → preprocess → detect → (maybe alert)
    second 31-60  → buffering ...
    second 60     → window ready → preprocess → detect → (maybe alert)
    ... and so on.
"""

import logging
import sys

import yaml

from model import load_model
from preprocessing import SampleBuffer, aggregate_window, load_scaler, normalise
from detector import detect, load_threshold
from mqtt_client import create_client, publish_alert

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# WHY: structured logging to stdout (not a file) because in Docker the
# standard practice is to let the container runtime collect stdout logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("anomaly-detector")


def main():
    """Load everything, connect to MQTT, and run forever."""

    # --- Step 1: Load configuration -------------------------------------------
    logger.info("Loading configuration from config.yaml ...")
    with open("config.yaml", "r") as f:
        # yaml.safe_load parses YAML into a Python dict.
        # WHY safe_load (not yaml.load): it refuses to execute arbitrary Python
        # code that could be embedded in a malicious YAML file.
        config = yaml.safe_load(f)

    mqtt_cfg = config["mqtt"]
    prep_cfg = config["preprocessing"]
    model_cfg = config["model"]

    # --- Step 2: Load trained artefacts ---------------------------------------
    logger.info("Loading model, scaler, and threshold ...")

    model = load_model(
        model_path=model_cfg["model_path"],
        use_npu=model_cfg.get("use_npu", False),
    )
    scaler = load_scaler(model_cfg["scaler_path"])
    threshold = load_threshold(model_cfg["threshold_path"])

    logger.info(f"Model loaded.  Anomaly threshold = {threshold:.6f}")

    # --- Step 3: Set up the sample buffer -------------------------------------
    buffer = SampleBuffer(
        window_size=prep_cfg["window_size"],
        tag_names=prep_cfg["tag_names"],
    )

    # --- Step 4: Define what happens when a sensor message arrives ------------
    def handle_sensor_message(sensor_data: dict) -> None:
        """
        Callback invoked by mqtt_client every time a message arrives.

        This is where the magic happens:
            raw message → buffer → window → aggregate → normalise → detect → alert
        """
        # Add the sample to the buffer
        window = buffer.add(sensor_data)

        if window is None:
            # Buffer not yet full — nothing to do, wait for more samples.
            return

        # --- Window is ready!  Process it. ------------------------------------
        # Step A: average the 30 samples into one vector of 10 values
        feature_vector = aggregate_window(window)

        # Step B: normalise to [0, 1] using the scaler fitted during training
        normalised = normalise(feature_vector, scaler)

        # Step C: run anomaly detection
        result = detect(model, normalised, threshold)

        # Step D: log the result and publish an alert if anomalous
        if result.is_anomaly:
            logger.warning(
                f"🚨 ANOMALY DETECTED  score={result.score:.6f}  "
                f"threshold={result.threshold:.6f}"
            )
        else:
            logger.info(
                f"✅ Normal  score={result.score:.6f}  "
                f"threshold={result.threshold:.6f}"
            )

        # Always publish the result (consumers can filter on is_anomaly)
        alert_json = result.to_alert_json()
        publish_alert(mqtt_client, mqtt_cfg["alert_topic"], alert_json)

    # --- Step 5: Connect to MQTT and start listening --------------------------
    logger.info("Connecting to MQTT broker ...")
    mqtt_client = create_client(
        broker_host=mqtt_cfg["broker_host"],
        broker_port=mqtt_cfg["broker_port"],
        client_id=mqtt_cfg["client_id"],
        sensor_topic=mqtt_cfg["sensor_topic"],
        on_sensor_message=handle_sensor_message,
    )

    # loop_forever() blocks and processes MQTT messages until the program is
    # killed.  In Docker, stopping the container sends SIGTERM which ends this.
    logger.info("Anomaly detector running.  Waiting for sensor data ...")
    mqtt_client.loop_forever()


if __name__ == "__main__":
    main()
