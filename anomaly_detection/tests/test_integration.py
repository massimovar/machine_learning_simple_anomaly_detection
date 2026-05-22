"""
test_integration.py — End-to-end integration test for the anomaly detection stack
===================================================================================

PURPOSE:
    This script verifies the full running system:
        sensor publisher (this script)
            → MQTT broker (Eclipse Mosquitto, port 1883)
                → anomaly detector (Docker container)
                    → alert subscriber (this script)

    It runs TWO scenarios:
        1. NORMAL  — publishes 30 samples with tag values inside the normal range.
           Expected result: is_anomaly == false.

        2. ANOMALY — publishes 30 samples with several tags pushed far outside
           the normal range (3–5× the noise width).
           Expected result: is_anomaly == true.

PREREQUISITES:
    - The stack must already be running:
          podman compose up -d
    - paho-mqtt must be installed in your local Python environment:
          pip install "paho-mqtt>=2.0,<3.0"

USAGE:
    python tests/test_integration.py                  # from anomaly_detection/
    pytest  tests/test_integration.py -v              # via pytest

EXIT CODES:
    0  — all assertions passed
    1  — at least one assertion failed or timeout occurred
"""

import json
import sys
import time
import threading
from typing import Optional

import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# Configuration — matches config.yaml
# ---------------------------------------------------------------------------
BROKER_HOST = "localhost"
BROKER_PORT = 1883
SENSOR_TOPIC = "ftoptix/paperclip/sensors"
ALERT_TOPIC = "anomaly/paperclip/alerts"
WINDOW_SIZE = 30          # detector buffers this many samples before detecting
MSG_INTERVAL_S = 0.05     # 50 ms between messages (30 msgs = 1.5 s per window)
ALERT_TIMEOUT_S = 30      # wait at most 30 s for a detection result


# ---------------------------------------------------------------------------
# Sensor data templates
# ---------------------------------------------------------------------------

def normal_payload() -> dict:
    """Return a flat dict of sensor values inside the normal operating range."""
    return {
        "wire_feed_speed":      10.0,   # normal: [8, 12] m/min
        "wire_tension":         22.5,   # normal: [15, 30] N
        "motor_current_feed":   2.25,   # normal: [1.5, 3.0] A
        "motor_current_bend":   3.25,   # normal: [2.0, 4.5] A
        "bend_angle":           180.0,  # normal: [178, 182] deg
        "cycle_time":           200.0,  # normal: [180, 220] ms
        "vibration_rms":        1.75,   # normal: [0.5, 3.0] mm/s
        "cut_force":            55.0,   # normal: [40, 70] N
        "temperature_motor":    47.5,   # normal: [30, 65] °C
        "reject_count_per_min": 1.0,    # normal: [0, 2] count
    }


def anomaly_payload() -> dict:
    """
    Return a flat dict of sensor values with several tags pushed far outside
    the normal range.  3-5× the noise width forces a high reconstruction error.
    """
    return {
        "wire_feed_speed":      35.0,   # ← 3.5× above normal max (12 m/min)
        "wire_tension":         90.0,   # ← far above normal max (30 N)
        "motor_current_feed":   12.0,   # ← far above normal max (3.0 A)
        "motor_current_bend":   16.0,   # ← far above normal max (4.5 A)
        "bend_angle":           180.0,  # within range
        "cycle_time":           200.0,  # within range
        "vibration_rms":        1.75,   # within range
        "cut_force":            55.0,   # within range
        "temperature_motor":    47.5,   # within range
        "reject_count_per_min": 1.0,    # within range
    }


# ---------------------------------------------------------------------------
# Alert receiver
# ---------------------------------------------------------------------------

class AlertReceiver:
    """
    Subscribes to the alert topic and stores the first message received.
    Thread-safe: the MQTT callback (background thread) sets an Event that the
    main thread waits on.
    """

    def __init__(self):
        self._result: Optional[dict] = None
        self._event = threading.Event()

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="test-alert-receiver",
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.connect(BROKER_HOST, BROKER_PORT)
        self._client.loop_start()   # runs the MQTT network loop in a background thread

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(ALERT_TOPIC)

    def _on_message(self, client, userdata, msg):
        try:
            self._result = json.loads(msg.payload.decode("utf-8"))
            self._event.set()   # wake up wait_for_alert()
        except json.JSONDecodeError:
            pass

    def wait_for_alert(self, timeout: float) -> Optional[dict]:
        """Block until an alert arrives or timeout expires.  Returns the alert dict or None."""
        self._event.wait(timeout=timeout)
        return self._result

    def reset(self):
        """Clear the last result so we can wait for the next window."""
        self._result = None
        self._event.clear()

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()


# ---------------------------------------------------------------------------
# Sensor publisher
# ---------------------------------------------------------------------------

def publish_window(sensor_data: dict, n_messages: int = WINDOW_SIZE):
    """
    Publish `n_messages` identical sensor readings on the sensor topic.
    Uses a fresh MQTT client per call (simple, no state to manage).

    Parameters
    ----------
    sensor_data : dict
        Flat sensor dict matching the detector's tag_names.
    n_messages : int
        Number of messages to publish (must fill at least one window = 30).
    """
    pub = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="test-sensor-publisher",
    )
    pub.connect(BROKER_HOST, BROKER_PORT)
    pub.loop_start()

    payload = json.dumps(sensor_data)
    for _ in range(n_messages):
        pub.publish(SENSOR_TOPIC, payload, qos=1)
        time.sleep(MSG_INTERVAL_S)

    pub.loop_stop()
    pub.disconnect()


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_result(result: Optional[dict], expect_anomaly: bool, scenario: str):
    """Raise AssertionError with a clear message if the result is unexpected."""
    if result is None:
        raise AssertionError(
            f"[{scenario}] TIMEOUT — no alert received within {ALERT_TIMEOUT_S} s.\n"
            "Is the detector container running?  Try: podman compose up -d"
        )

    actual = result.get("is_anomaly")
    if actual != expect_anomaly:
        raise AssertionError(
            f"[{scenario}] Expected is_anomaly={expect_anomaly}, "
            f"got is_anomaly={actual}.\n"
            f"Full alert: {result}"
        )

    status = "ANOMALY DETECTED ✓" if actual else "NORMAL ✓"
    print(
        f"  [{scenario}] PASS — {status}  "
        f"score={result.get('anomaly_score'):.6f}  "
        f"threshold={result.get('threshold'):.6f}"
    )


# ---------------------------------------------------------------------------
# Test cases (runnable both standalone and via pytest)
# ---------------------------------------------------------------------------

def test_normal_window():
    """Publishing 30 normal samples must NOT trigger an anomaly."""
    receiver = AlertReceiver()
    receiver.reset()           # flush any stale message from a previous run

    print("  [NORMAL] Publishing 30 normal sensor samples ...")
    publish_window(normal_payload())

    result = receiver.wait_for_alert(timeout=ALERT_TIMEOUT_S)
    receiver.stop()

    assert_result(result, expect_anomaly=False, scenario="NORMAL")


def test_anomaly_window():
    """Publishing 30 anomalous samples MUST trigger an anomaly."""
    receiver = AlertReceiver()
    receiver.reset()

    print("  [ANOMALY] Publishing 30 anomalous sensor samples ...")
    publish_window(anomaly_payload())

    result = receiver.wait_for_alert(timeout=ALERT_TIMEOUT_S)
    receiver.stop()

    assert_result(result, expect_anomaly=True, scenario="ANOMALY")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Anomaly Detection — Integration Test")
    print(f"  Broker: {BROKER_HOST}:{BROKER_PORT}")
    print(f"  Sensor topic : {SENSOR_TOPIC}")
    print(f"  Alert  topic : {ALERT_TOPIC}")
    print(f"  Window size  : {WINDOW_SIZE} messages")
    print("=" * 60)

    failures = []

    for name, fn in [("Normal window", test_normal_window),
                     ("Anomaly window", test_anomaly_window)]:
        print(f"\n[TEST] {name}")
        try:
            fn()
        except AssertionError as exc:
            print(f"  FAIL — {exc}")
            failures.append(name)
        except Exception as exc:
            print(f"  ERROR — {exc}")
            failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print(f"  RESULT: FAILED ({len(failures)} test(s) failed: {', '.join(failures)})")
        print("=" * 60)
        sys.exit(1)
    else:
        print("  RESULT: ALL TESTS PASSED ✓")
        print("=" * 60)
        sys.exit(0)
