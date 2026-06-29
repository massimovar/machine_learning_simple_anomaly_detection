import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import PreprocessingError, SampleBuffer, aggregate_window, normalise  # noqa: E402


class RecordingScaler:
    def __init__(self):
        self.seen = None

    def transform(self, values):
        self.seen = values.copy()
        return values / 10.0


class PreprocessingTests(unittest.TestCase):
    def test_sample_buffer_uses_configured_sensor_tag_order(self):
        buffer = SampleBuffer(window_size=2, tag_names=["b", "a"])

        self.assertIsNone(buffer.add({"a": 1, "b": 2}))
        window = buffer.add({"a": 3, "b": 4})

        np.testing.assert_array_equal(
            window,
            np.array([[2, 1], [4, 3]], dtype=np.float32),
        )

    def test_sample_buffer_reports_missing_sensor_tags(self):
        buffer = SampleBuffer(window_size=1, tag_names=["a", "b"])

        with self.assertRaisesRegex(PreprocessingError, "missing sensor tag: b"):
            buffer.add({"a": 1})

    def test_sample_buffer_reports_non_numeric_sensor_values(self):
        buffer = SampleBuffer(window_size=1, tag_names=["a"])

        with self.assertRaisesRegex(PreprocessingError, "sensor values must be numeric"):
            buffer.add({"a": "bad"})

    def test_aggregate_window_averages_each_tag(self):
        result = aggregate_window(np.array([[1, 3], [5, 7]], dtype=np.float32))

        np.testing.assert_array_equal(result, np.array([3, 5], dtype=np.float32))

    def test_aggregate_window_rejects_wrong_shape(self):
        with self.assertRaisesRegex(PreprocessingError, "non-empty 2-D array"):
            aggregate_window(np.array([1, 2, 3]))

    def test_normalise_uses_scaler(self):
        scaler = RecordingScaler()

        result = normalise(np.array([2, 4], dtype=np.float32), scaler)

        np.testing.assert_array_equal(scaler.seen, np.array([[2, 4]], dtype=np.float32))
        np.testing.assert_array_equal(result, np.array([0.2, 0.4], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
