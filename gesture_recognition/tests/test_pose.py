"""Tests for pose.py — preprocessing math and a real-model smoke test."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pose  # noqa: E402
from pose import PoseEstimator, center_square, preprocess  # noqa: E402

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "movenet_lightning_int8.tflite")


class TestPreprocessing(unittest.TestCase):

    def test_center_square_of_landscape_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[0, 80] = 255                       # top-left corner of the centred square
        crop = center_square(frame)
        self.assertEqual(crop.shape, (480, 480, 3))
        self.assertEqual(crop[0, 0, 0], 255)

    def test_center_square_of_portrait_frame(self):
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        frame[80, 0] = 255
        crop = center_square(frame)
        self.assertEqual(crop.shape, (480, 480, 3))
        self.assertEqual(crop[0, 0, 0], 255)

    def test_preprocess_gives_uint8_rgb_batch(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)                # pure blue in OpenCV's BGR order
        batch = preprocess(frame, 192, 192)
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
