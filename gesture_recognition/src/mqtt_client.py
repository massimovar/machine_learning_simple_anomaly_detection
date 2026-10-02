"""
mqtt_client.py — publish gesture events and service health to Mosquitto
========================================================================

WHY THIS FILE EXISTS:
    FT Optix learns about gestures only through MQTT.  Two topics are used:

      gesture/paperclip/events   one message per gesture
                                 QoS 1: delivered at least once
                                 not retained: never replayed when Optix restarts
      gesture/paperclip/status   heartbeat every 2 s and on every state change
                                 retained: a new subscriber sees the state at once
                                 last will "offline": the broker announces a crash

    Every event carries a unique event_id (so FT Optix can ignore a QoS 1
    duplicate) and a UTC timestamp (so it can ignore stale events).
"""

import json
import logging
import uuid
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
OFFLINE = {"v": SCHEMA_VERSION, "state": "offline"}


def utc_timestamp():
    """ISO-8601 UTC with milliseconds, e.g. 2026-10-02T12:00:00.123Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_event(gesture, confidence):
    return {"v": SCHEMA_VERSION, "event_id": str(uuid.uuid4()), "gesture": gesture,
            "confidence": confidence, "timestamp": utc_timestamp()}


def make_status(state, person_present=False, fps=0.0, inference_ms=0.0, accelerator="cpu"):
    return {"v": SCHEMA_VERSION, "state": state, "person_present": person_present,
            "fps": round(fps, 1), "inference_ms": round(inference_ms, 1),
            "accelerator": accelerator, "timestamp": utc_timestamp()}


class GesturePublisher:
    def __init__(self, host, port, client_id, events_topic, status_topic, client=None):
        self._host, self._port = host, port
        self._events_topic = events_topic
        self._status_topic = status_topic
        self._last_status = None
        self._client = client or mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        # WHY: if this process dies or its network drops, the broker itself
        # publishes "offline" on the status topic (Last Will and Testament).
        self._client.will_set(status_topic, json.dumps(OFFLINE), qos=1, retain=True)
        self._client.reconnect_delay_set(min_delay=1, max_delay=10)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def start(self):
        # WHY connect_async + loop_start: the service keeps running, and keeps
        # retrying in the background, even if the broker is not up yet.
        self._client.connect_async(self._host, self._port, keepalive=15)
        self._client.loop_start()

    def publish_event(self, gesture, confidence):
        """Publish one gesture; returns the payload, or None if dropped."""
        if not self._client.is_connected():
            # WHY drop instead of queue: a gesture delivered seconds late would
            # surprise the operator (FT Optix discards stale events anyway).
            logger.warning("MQTT offline: dropped %s event", gesture)
            return None
        event = make_event(gesture, confidence)
        self._client.publish(self._events_topic, json.dumps(event), qos=1, retain=False)
        logger.info("Gesture %s (confidence %.2f) id=%s", gesture, confidence, event["event_id"])
        return event

    def publish_status(self, status):
        self._last_status = status
        if self._client.is_connected():
            self._client.publish(self._status_topic, json.dumps(status), qos=1, retain=True)

    def stop(self, final_status):
        """Publish a clean final status (normally "offline"), then disconnect."""
        try:
            if self._client.is_connected():
                info = self._client.publish(self._status_topic, json.dumps(final_status), qos=1, retain=True)
                info.wait_for_publish(timeout=2)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Final status not delivered: %s", exc)
        finally:
            self._client.disconnect()
            self._client.loop_stop()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info("Connected to MQTT broker %s:%d", self._host, self._port)
            # WHY: replaces a retained "offline" left by the last will at once,
            # instead of waiting for the next heartbeat.
            if self._last_status is not None:
                status = dict(self._last_status, timestamp=utc_timestamp())
                client.publish(self._status_topic, json.dumps(status), qos=1, retain=True)
        else:
            logger.error("MQTT connection refused: %s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            logger.warning("MQTT disconnected (%s); reconnecting", reason_code)
