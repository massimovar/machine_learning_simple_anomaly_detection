"""
mqtt_client.py — connect, subscribe, publish

Topics and payloads are documented in the README ("Evidence over MQTT").
FactoryTalk Optix reads these payloads field by field: keep their keys stable.
"""

import json
import logging

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


def flatten_records(payload: dict) -> dict:
    """FTOptix's {"Records": [{"TagName": t, "Value": v}, ...]} -> {t: v}.

    A flat payload (e.g. from mosquitto_pub or tests/test_integration.py) passes through.
    """
    records = payload.get("Records")
    if not isinstance(records, list):
        return payload
    return {r["TagName"]: r["Value"] for r in records}


def publish(client: mqtt.Client, topic: str, payload: dict, retain: bool = False) -> None:
    # qos=1 (at least once): an alert must not be lost silently; qos=2 is not needed here.
    client.publish(topic, json.dumps(payload), qos=1, retain=retain)
    logger.info("Published to %s", topic)


def create_client(broker_host: str, broker_port: int, client_id: str,
                  sensor_topic: str, on_sensor_message,
                  command_topic: str, on_command,
                  status_topic: str, status_payload) -> mqtt.Client:
    """Connect to the broker; on every (re)connect subscribe again and republish the status.

    The status is retained, so an HMI that connects later still sees it; the
    Last Will makes the broker replace it with {"state": "offline"} if this
    process vanishes.
    """
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    client.will_set(status_topic, json.dumps({"state": "offline"}), qos=1, retain=True)

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            logger.error("Connection failed with code: %s", reason_code)
            return
        logger.info("Connected to MQTT broker at %s:%s", broker_host, broker_port)
        client.subscribe(sensor_topic)
        client.subscribe(command_topic)
        publish(client, status_topic, status_payload(), retain=True)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            if msg.topic == command_topic:
                on_command(payload)
            else:
                on_sensor_message(flatten_records(payload))
        except Exception as exc:
            logger.error("Dropped message on %s (%r): %s", msg.topic, exc, msg.payload[:100])

    client.on_connect = on_connect
    client.on_message = on_message
    logger.info("Connecting to MQTT broker %s:%s ...", broker_host, broker_port)
    client.connect(broker_host, broker_port)
    return client
