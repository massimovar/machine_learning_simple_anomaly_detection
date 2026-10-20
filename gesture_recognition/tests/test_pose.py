"""Tests for pose.py — preprocessing math and a real-model smoke test."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pose  # noqa: E402
from pose import PoseEstimator, letterbox, preprocess  # noqa: E402

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "movenet_lightning_int8.tflite")


class TestPreprocessing(unittest.TestCase):

    def test_letterbox_of_landscape_frame_pads_top_and_bottom(self):
        frame = np.full((360, 640, 3), 255, dtype=np.uint8)
        square = letterbox(frame, 192)
        self.assertEqual(square.shape, (192, 192, 3))
        # 640 -> 192 (x0.3), so 360 -> 108 rows, centred: rows 42..149
        self.assertEqual(square[41, 96, 0], 0)
        self.assertEqual(square[42, 96, 0], 255)
        self.assertEqual(square[149, 96, 0], 255)
        self.assertEqual(square[150, 96, 0], 0)
        self.assertEqual(square[96, 0, 0], 255)       # full width kept
        self.assertEqual(square[96, 191, 0], 255)

    def test_letterbox_of_portrait_frame_pads_left_and_right(self):
        frame = np.full((640, 480, 3), 255, dtype=np.uint8)
        square = letterbox(frame, 192)
        # 480 -> 144 columns, centred: columns 24..167
        self.assertEqual(square[96, 23, 0], 0)
        self.assertEqual(square[96, 24, 0], 255)
        self.assertEqual(square[96, 167, 0], 255)
        self.assertEqual(square[96, 168, 0], 0)

    def test_preprocess_gives_uint8_rgb_batch(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)                # pure blue in OpenCV's BGR order
        batch = preprocess(frame, 192)
        self.assertEqual(batch.shape, (1, 192, 192, 3))
        self.assertEqual(batch.dtype, np.uint8)
        self.assertEqual(tuple(batch[0, 96, 96]), (0, 0, 255))   # blue is last in RGB


@unittest.skipUnless(os.path.exists(MODEL), "run tools/fetch_model.py first")
class TestRealModel(unittest.TestCase):

    def test_cpu_inference_returns_17_keypoints(self):
        estimator = PoseEstimator(MODEL)
        keypoints = estimator.infer(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(estimator.accelerator, "cpu")
        self.assertEqual(keypoints.shape, (17, 3))
        self.assertEqual(keypoints.dtype, np.float32)
        self.assertTrue(np.all((keypoints >= 0) & (keypoints <= 1)))

    def test_missing_npu_delegate_falls_back_to_cpu(self):
        with self.assertLogs("pose", level="WARNING"):
            estimator = PoseEstimator(MODEL, use_npu=True, vx_delegate_path="/nonexistent/libvx_delegate.so")
        self.assertEqual(estimator.accelerator, "cpu")
        self.assertEqual(estimator.infer(np.zeros((360, 640, 3), dtype=np.uint8)).shape, (17, 3))

    def test_npu_failing_at_first_inference_falls_back_to_cpu(self):
        # The delegate loads, but the NPU fails when it compiles the graph,
        # which happens at the first inference.
        class NpuFailsOnInvoke(pose.Interpreter):
            def __init__(self, *args, experimental_delegates=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.on_npu = bool(experimental_delegates)

            def invoke(self):
                if self.on_npu:
                    raise RuntimeError("graph compilation failed")
                super().invoke()

        with mock.patch.object(pose, "Interpreter", NpuFailsOnInvoke), \
                mock.patch.object(pose, "load_delegate", lambda path: object()), \
                self.assertLogs("pose", level="WARNING"):
            estimator = PoseEstimator(MODEL, use_npu=True)
            self.assertEqual(estimator.accelerator, "cpu")
            self.assertEqual(estimator.infer(np.zeros((360, 640, 3), dtype=np.uint8)).shape, (17, 3))


if __name__ == "__main__":
    unittest.main()
