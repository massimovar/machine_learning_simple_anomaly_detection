import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mqtt_client import flatten_records  # noqa: E402
from preprocessing import SampleBuffer, load_scaler, normalise  # noqa: E402


class PreprocessingTests(unittest.TestCase):
    def test_buffer_returns_the_per_tag_mean_in_configured_order(self):
        buffer = SampleBuffer(window_size=2, tag_names=["b", "a"])
        self.assertIsNone(buffer.add({"a": 1, "b": 2}))
        np.testing.assert_array_equal(buffer.add({"a": 3, "b": 4}), np.array([3, 2], dtype=np.float32))
        self.assertIsNone(buffer.add({"a": 5, "b": 6}))  # a new window starts

    def test_bad_messages_are_rejected_and_not_buffered(self):
        buffer = SampleBuffer(window_size=1, tag_names=["a", "b"])
        with self.assertRaises(KeyError):
            buffer.add({"a": 1})
        with self.assertRaises(ValueError):
            buffer.add({"a": "bad", "b": 1})
        self.assertEqual(buffer.rows, [])

    def test_normalise_maps_the_training_range_to_0_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scaler.json")
            with open(path, "w") as f:
                json.dump({"min": [0, 10], "max": [2, 20]}, f)
            result = normalise(np.array([1, 25], dtype=np.float32), load_scaler(path))
        np.testing.assert_allclose(result, [0.5, 1.5])
        self.assertEqual(result.dtype, np.float32)

    def test_ftoptix_records_are_flattened_and_flat_payloads_pass(self):
        records = {"Records": [{"TagName": "a", "Value": 1.5}, {"TagName": "b", "Value": 2}], "Timestamp": "t"}
        self.assertEqual(flatten_records(records), {"a": 1.5, "b": 2})
        self.assertEqual(flatten_records({"a": 1}), {"a": 1})


if __name__ == "__main__":
    unittest.main()
