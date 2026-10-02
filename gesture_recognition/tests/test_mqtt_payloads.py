"""Tests for mqtt_client.py — payload schema and publish rules, with a fake client."""

import json
import os
import re
import sys
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mqtt_client import GesturePublisher, make_event, make_status  # noqa: E402

TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
EVENTS, STATUS = "gesture/paperclip/events", "gesture/paperclip/status"


class FakeInfo:
    def wait_for_publish(self, timeout=None):
        pass


class FakeClient:
    """Records what the publisher does instead of talking to a broker."""

    def __init__(self):
        self.connected = False
        self.will = None
        self.published = []
        self.calls = []

    def will_set(self, topic, payload, qos, retain):
        self.will = (topic, json.loads(payload), qos, retain)

    def reconnect_delay_set(self, min_delay, max_delay):
        pass

    def connect_async(self, host, port, keepalive):
        self.calls.append("connect_async")

    def loop_start(self):
        self.calls.append("loop_start")

    def loop_stop(self):
        self.calls.append("loop_stop")

    def disconnect(self):
        self.calls.append("disconnect")

    def is_connected(self):
        return self.connected

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, json.loads(payload), qos, retain))
        return FakeInfo()


class TestPayloads(unittest.TestCase):

    def test_event_schema(self):
        event = make_event("swipe_left", 0.92)
        self.assertEqual(set(event), {"v", "event_id", "gesture", "confidence", "timestamp"})
        self.assertEqual(event["v"], 1)
        self.assertEqual((event["gesture"], event["confidence"]), ("swipe_left", 0.92))
        uuid.UUID(event["event_id"])
        self.assertRegex(event["timestamp"], TIMESTAMP)

    def test_every_event_has_a_unique_id(self):
        self.assertNotEqual(make_event("palm_up", 0.8)["event_id"], make_event("palm_up", 0.8)["event_id"])

    def test_status_schema_rounds_metrics(self):
        status = make_status("running", True, fps=19.847, inference_ms=18.4321, accelerator="npu")
        self.assertEqual(set(status), {"v", "state", "person_present", "fps", "inference_ms",
                                       "accelerator", "timestamp"})
        self.assertEqual((status["fps"], status["inference_ms"]), (19.8, 18.4))
        self.assertRegex(status["timestamp"], TIMESTAMP)


class TestPublisher(unittest.TestCase):

    def setUp(self):
        self.client = FakeClient()
        self.publisher = GesturePublisher("broker", 1883, "test", EVENTS, STATUS, client=self.client)

    def test_last_will_is_retained_offline(self):
        self.assertEqual(self.client.will, (STATUS, {"v": 1, "state": "offline"}, 1, True))

    def test_event_is_dropped_while_disconnected(self):
        self.assertIsNone(self.publisher.publish_event("swipe_right", 0.9))
        self.assertEqual(self.client.published, [])

    def test_event_is_qos1_and_not_retained(self):
        self.client.connected = True
        self.publisher.publish_event("swipe_right", 0.9)
        topic, payload, qos, retain = self.client.published[0]
        self.assertEqual((topic, payload["gesture"], qos, retain), (EVENTS, "swipe_right", 1, False))

    def test_status_is_retained_and_resent_on_reconnect(self):
        self.publisher.publish_status(make_status("running"))
        self.assertEqual(self.client.published, [], "nothing can be sent while disconnected")
        self.client.connected = True
        self.client.on_connect(self.client, None, None, 0, None)
        topic, payload, qos, retain = self.client.published[0]
        self.assertEqual((topic, payload["state"], qos, retain), (STATUS, "running", 1, True))

    def test_stop_sends_final_status_then_disconnects(self):
        self.client.connected = True
        self.publisher.stop(make_status("offline"))
        self.assertEqual(self.client.published[-1][1]["state"], "offline")
        self.assertEqual(self.client.calls[-2:], ["disconnect", "loop_stop"])


if __name__ == "__main__":
    unittest.main()
