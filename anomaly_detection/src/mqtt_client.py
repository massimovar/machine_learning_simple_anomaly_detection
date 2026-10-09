"""
mqtt_client.py — MQTT subscribe and publish helpers
====================================================

WHY THIS FILE EXISTS:
    MQTT is a lightweight publish/subscribe messaging protocol widely used in
    industrial IoT.  This file wraps the `paho-mqtt` library into a few small
    helpers:

    - create_client()   → connects, subscribes to sensor data (and the command
                          topic), sets the Last Will, publishes the status
    - publish_alert()   → sends an anomaly alert back to the broker
    - publish_status()  → (re)publishes the retained engine status
    - publish_json()    → publishes any dict, e.g. a benchmark result

WHAT IS MQTT? (quick primer)
    Think of it like a post office:
    - **Broker**: the post office.  It receives messages and routes them.
    - **Publisher**: someone dropping off a letter (e.g., FTOptix publishing sensor data).
    - **Subscriber**: someone picking up letters for a topic (e.g., this anomaly detector).
    - **Topic**: the "address" on the letter (e.g., "ftoptix/paperclip/sensors").

    In this project:
        FTOptix  ──publishes──►  MQTT Broker  ──delivers──►  Our detector
        Our detector  ──publishes──►  MQTT Broker  ──delivers──►  (any listener)

MQTT TOPICS:
    - Sensor data (incoming): configured in config.yaml as mqtt.sensor_topic
      Default: "ftoptix/paperclip/sensors"
    - Alert results (outgoing): configured in config.yaml as mqtt.alert_topic
      Default: "anomaly/paperclip/alerts"

==============================================================================
EXPECTED INCOMING PAYLOAD (sensor data)
==============================================================================

    Published by FTOptix to topic "ftoptix/paperclip/sensors".
    FTOptix publishes data in its native "Records" envelope format:

    {
        "Records": [
            { "TagName": "wire_feed_speed",      "Value": 10.2   },
            { "TagName": "wire_tension",          "Value": 22.5   },
            { "TagName": "motor_current_feed",    "Value": 2.1    },
            { "TagName": "motor_current_bend",    "Value": 3.3    },
            { "TagName": "bend_angle",            "Value": 180.1  },
            { "TagName": "cycle_time",            "Value": 198    },
            { "TagName": "vibration_rms",         "Value": 1.8    },
            { "TagName": "cut_force",             "Value": 55.0   },
            { "TagName": "temperature_motor",     "Value": 48.3   },
            { "TagName": "reject_count_per_min",  "Value": 1      }
        ],
        "Timestamp": "2026-02-18T15:29:42.5760812"
    }

    This module automatically unwraps the Records envelope into a flat dict:
        {"wire_feed_speed": 10.2, "wire_tension": 22.5, ...}
    which is what the rest of the pipeline (preprocessing, model, detector) expects.

    A flat JSON payload (without the Records wrapper) is also accepted for
    testing with tools like mosquitto_pub.

    Tag reference (NOT part of the JSON — for developers only):

    Tag                     Type    Unit     Normal range
    ─────────────────────── ─────── ──────── ────────────
    wire_feed_speed         float   m/min    [8, 12]
    wire_tension            float   N        [15, 30]
    motor_current_feed      float   A        [1.5, 3.0]
    motor_current_bend      float   A        [2.0, 4.5]
    bend_angle              float   degrees  [178, 182]
    cycle_time              float   ms       [180, 220]
    vibration_rms           float   mm/s     [0.5, 3.0]
    cut_force               float   N        [40, 70]
    temperature_motor       float   °C       [30, 65]
    reject_count_per_min    int     count    [0, 2]

    NOTES:
    - The 10 tag keys must match the tag_names list in config.yaml (order matters).
    - Values are raw (un-normalised). The detector normalises them internally
      using the MinMaxScaler fitted during training.
    - Messages arrive at 1 Hz (one per second). The detector buffers 30 messages
      (one window) before running a detection cycle.

==============================================================================
OUTGOING ALERT PAYLOAD (detection result)
==============================================================================

    Published by this detector to topic "anomaly/paperclip/alerts" after
    every completed detection window (every 30 seconds at 1 Hz).

    Example (copy-paste-ready JSON):

    {
        "is_anomaly":    true,
        "anomaly_score": 0.042871,
        "threshold":     0.010236,
        "accelerator":   "npu",
        "precision":     "int8",
        "inference_ms":  0.4123
    }

    Field reference (NOT part of the JSON — for developers only):

    Field           Type    Description
    ─────────────── ─────── ──────────────────────────────────────────────
    is_anomaly      bool    true if anomaly_score > threshold
    anomaly_score   float   MSE between input and reconstruction (6 d.p.)
    threshold       float   Three-sigma boundary of the active model variant (6 d.p.)
    accelerator     string  "npu" or "cpu": where this inference really ran
    precision       string  "int8" or "float32": which model variant ran
    inference_ms    float   Duration of this single invoke() call

    The first three fields are unchanged since V1; the evidence fields are
    appended, so older consumers that read only "is_anomaly" keep working.

    INTERPRETING THE RESULT:
    - is_anomaly == false, score << threshold  → machine is operating normally
    - is_anomaly == false, score close to threshold → borderline, worth monitoring
    - is_anomaly == true,  score >> threshold  → clear anomaly, investigate

    NOTES:
    - An alert is published for EVERY detection window, not only for anomalies.
      Consumers can filter on the "is_anomaly" field.
    - QoS 1 (at least once delivery) is used to avoid losing anomaly alerts.

==============================================================================
STATUS, COMMAND AND BENCHMARK TOPICS (NPU evidence)
==============================================================================

    Status — mqtt.status_topic (default "anomaly/paperclip/status"), RETAINED,
    published on every (re)connect and after each benchmark. A late subscriber
    (e.g. FTOptix starting after the detector) gets it immediately.

    {
        "state": "online", "started_at": "2026-10-08T10:00:00+00:00",
        "accelerator": "npu", "npu_requested": true, "npu_error": null,
        "model": "autoencoder_int8.tflite", "precision": "int8",
        "runtime": "tflite_runtime 2.16.2", "delegate": "/opt/npu/lib/libvx_delegate.so",
        "delegation": [{"delegate": "Vx Delegate", "nodes": 5, "of": 5}],
        "warmup_ms": 66.0, "threshold": 0.010236,
        "galcore_version": "libGAL 6.4.11:745085",
        "graph_cache": {"enabled": true, "dir": "/var/cache/npu", "files": 1,
                        "bytes": 54976, "names": ["<hash>.nb"]}
    }

    If the process dies, the broker publishes the Last Will {"state": "offline"}
    (retained) on the same topic, so a stale "online" badge cannot survive.

    Command — mqtt.command_topic (default "anomaly/paperclip/cmd"), consumed:
        {"cmd": "benchmark", "runs": 200}       runs is capped at 5000

    Benchmark — mqtt.benchmark_topic (default "anomaly/paperclip/benchmark"):
    the same model and the same input, timed on the CPU and then on the active engine.

    {
        "model": "autoencoder_int8.tflite", "precision": "int8", "runs": 5000,
        "cpu": {"accelerator": "cpu", "runs": 5000, "p50_ms": 0.3503, "p95_ms": 0.5473, ...},
        "npu": {"accelerator": "npu", "runs": 5000, "p50_ms": 0.6555, "p95_ms": 0.754, ...},
        "npu_unavailable": null,
        "max_output_difference": 0.0,
        "counters_before": null, "counters_after": null,
        "npu_vs_cpu_p50": 1.87
    }

    "npu" is null and "npu_unavailable" holds the reason when the NPU is not
    in use. "counters_*" are null unless NPU_COUNTERS_PATH points to readable
    driver counters. All numbers above are illustrative, not measured.
"""

import json
import logging
from typing import Callable, Optional

import paho.mqtt.client as mqtt

# WHY: We use Python's built-in logging instead of print() so messages have
# timestamps and severity levels, making debugging easier.
logger = logging.getLogger(__name__)


def _flatten_ftoptix_records(payload: dict) -> dict:
    """Convert FTOptix Records envelope to a flat sensor dict when present."""
    records = payload.get("Records")
    if not isinstance(records, list):
        return payload

    flat = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        tag = record.get("TagName", record.get("tagName"))
        value = record.get("Value", record.get("value"))
        if tag is not None and value is not None:
            flat[tag] = value
    return flat


def create_client(
    broker_host: str,
    broker_port: int,
    client_id: str,
    sensor_topic: str,
    on_sensor_message: Callable[[dict], None],
    status_topic: Optional[str] = None,
    status_payload: Optional[Callable[[], dict]] = None,
    command_topic: Optional[str] = None,
    on_command: Optional[Callable[[dict], None]] = None,
) -> mqtt.Client:
    """
    Create an MQTT client, connect to the broker, and subscribe to sensor data.

    Parameters
    ----------
    broker_host : str
        Hostname or IP of the MQTT broker (e.g., "mqtt-broker" in Docker).
    broker_port : int
        Port number (typically 1883 for unencrypted MQTT).
    client_id : str
        A unique name for this client.
    sensor_topic : str
        The MQTT topic to subscribe to (e.g., "ftoptix/paperclip/sensors").
    on_sensor_message : callable
        A function that will be called with the parsed JSON dict every time
        a sensor message arrives.
    status_topic, status_payload : optional
        Retained "who am I and where do I run" message. status_payload() is
        called on every (re)connect, so a restarted broker gets it again.
        The broker publishes {"state": "offline"} there if we vanish
        (MQTT Last Will).
    command_topic, on_command : optional
        Requests from the HMI (e.g. {"cmd": "benchmark", "runs": 1000}).

    Returns
    -------
    mqtt.Client
        The connected MQTT client (call .loop_start() or .loop_forever() to run).
    """

    # --- Create the client ----------------------------------------------------
    # paho-mqtt v2 requires specifying the callback API version.
    # CallbackAPIVersion.VERSION2 is the modern style.
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
    )
    if status_topic:
        # WHY retained + Last Will: an HMI that connects later still sees the
        # current status, and sees "offline" if the detector dies.
        client.will_set(status_topic, json.dumps({"state": "offline"}), qos=1, retain=True)

    # --- Define callbacks (what happens on certain MQTT events) ---------------

    def on_connect(client, userdata, flags, reason_code, properties):
        """Called when the client successfully connects to the broker."""
        if reason_code == 0:
            logger.info(f"Connected to MQTT broker at {broker_host}:{broker_port}")
            # Subscribe to the sensor topic right after connecting.
            # WHY subscribe here (not outside): if the connection drops and
            # reconnects, the subscription is automatically renewed.
            client.subscribe(sensor_topic)
            logger.info(f"Subscribed to topic: {sensor_topic}")
            if command_topic and on_command:
                client.subscribe(command_topic)
                logger.info(f"Subscribed to topic: {command_topic}")
            if status_topic and status_payload:
                publish_status(client, status_topic, status_payload())
        else:
            logger.error(f"Connection failed with code: {reason_code}")

    def on_message(client, userdata, msg):
        """Called every time a message arrives on a subscribed topic."""
        try:
            # msg.payload is raw bytes → decode to string → parse as JSON
            payload = json.loads(msg.payload.decode("utf-8"))
            if command_topic and msg.topic == command_topic:
                on_command(payload)
                return

            # Hand the flat dict to the callback provided by main.py
            on_sensor_message(_flatten_ftoptix_records(payload))
        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON message on {msg.topic}: {msg.payload[:100]}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    # Attach the callbacks to the client
    client.on_connect = on_connect
    client.on_message = on_message

    # --- Connect to the broker ------------------------------------------------
    logger.info(f"Connecting to MQTT broker {broker_host}:{broker_port} ...")
    client.connect(broker_host, broker_port)

    return client


def publish_alert(client: mqtt.Client, alert_topic: str, alert_json: str) -> None:
    """
    Publish an anomaly alert message to the MQTT broker.

    Parameters
    ----------
    client : mqtt.Client
        The connected MQTT client.
    alert_topic : str
        Topic to publish to (e.g., "anomaly/paperclip/alerts").
    alert_json : str
        The JSON string to publish (produced by DetectionResult.to_alert_json()).
    """
    # qos=1 means "at least once delivery" — the broker acknowledges receipt.
    # WHY qos=1: we don't want to silently lose anomaly alerts, but qos=2
    # ("exactly once") adds overhead we don't need for this use case.
    client.publish(alert_topic, alert_json, qos=1)
    logger.info(f"Published alert to {alert_topic}: {alert_json}")


def publish_status(client: mqtt.Client, topic: str, payload: dict) -> None:
    """Retained status: the broker keeps the last one for late subscribers."""
    client.publish(topic, json.dumps(payload), qos=1, retain=True)
    logger.info(f"Published status to {topic}: accelerator={payload.get('accelerator')}")


def publish_json(client: mqtt.Client, topic: str, payload: dict) -> None:
    client.publish(topic, json.dumps(payload), qos=1)
    logger.info(f"Published to {topic}")
