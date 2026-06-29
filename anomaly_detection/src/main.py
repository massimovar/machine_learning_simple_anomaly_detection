import logging
import sys

import yaml

from detector import detect, load_threshold
from model import load_tflite_model
from mqtt_client import create_client, publish_alert
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

    model = load_tflite_model(model_cfg["model_path"], use_npu=model_cfg.get("use_npu", False))
    scaler = load_scaler(model_cfg["scaler_path"])
    threshold = load_threshold(model_cfg["threshold_path"])
    buffer = SampleBuffer(prep_cfg["window_size"], prep_cfg["tag_names"])

    def handle_sensor_message(sensor_data: dict) -> None:
        window = buffer.add(sensor_data)
        if window is None:
            return

        feature_vector = aggregate_window(window)
        normalised = normalise(feature_vector, scaler)
        result = detect(model, normalised, threshold)

        message = f"score={result.score:.6f} threshold={result.threshold:.6f}"
        if result.is_anomaly:
            logger.warning("Anomaly detected: %s", message)
        else:
            logger.info("Normal: %s", message)

        publish_alert(mqtt_client, mqtt_cfg["alert_topic"], result.to_alert_json())

    mqtt_client = create_client(
        broker_host=mqtt_cfg["broker_host"],
        broker_port=mqtt_cfg["broker_port"],
        client_id=mqtt_cfg["client_id"],
        sensor_topic=mqtt_cfg["sensor_topic"],
        on_sensor_message=handle_sensor_message,
    )

    logger.info("Anomaly detector running")
    mqtt_client.loop_forever()


if __name__ == "__main__":
    main()
