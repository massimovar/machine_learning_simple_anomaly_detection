"""Tests for camera.py — latest-frame slot and reconnect, with a fake capture."""

import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camera import CAMERA_LOST, RUNNING, STARTING, Camera, redact  # noqa: E402


class FakeCapture:
    """Delivers `frames` numbered frames, then fails (or blocks until released)."""

    def __init__(self, frames=0, opened=True, hold=None):
        self.frames, self.opened, self.hold = frames, opened, hold
        self.count = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.count < self.frames:
            self.count += 1
            return True, np.full((4, 4, 3), self.count, dtype=np.uint8)
        if self.hold is not None:
            self.hold.wait()
        return False, None

    def get(self, prop):
        return 0.0

    def release(self):
        self.released = True


class TestCamera(unittest.TestCase):

    def setUp(self):
        self.hold = threading.Event()

    def tearDown(self):
        self.hold.set()

    def make_camera(self, captures, states_at_open=None):
        captures = list(captures)

        def open_fn(source, open_timeout_ms, read_timeout_ms):
            if states_at_open is not None:
                states_at_open.append(camera.state)
            return captures.pop(0) if captures else FakeCapture(opened=False)

        camera = Camera("http://cam/stream.mjpg", reconnect_min_s=0.01, reconnect_max_s=0.02,
                        open_fn=open_fn)
        self.addCleanup(camera.stop)
        return camera

    def test_latest_returns_only_the_newest_frame(self):
        camera = self.make_camera([FakeCapture(frames=10, hold=self.hold)])
        camera.start()
        self.assertIsNotNone(camera.latest(9, timeout=2.0), "10 frames expected")
        frame, _, seq = camera.latest(0, timeout=0.1)
        self.assertEqual(seq, 10)
        self.assertEqual(frame[0, 0, 0], 10, "older frames must be dropped")
        self.assertIsNone(camera.latest(10, timeout=0.05), "no newer frame yet")
        self.assertEqual(camera.state, RUNNING)

    def test_lost_camera_reconnects_and_recovers(self):
        states = []
        first, broken = FakeCapture(frames=3), FakeCapture(opened=False)
        camera = self.make_camera([first, broken, FakeCapture(frames=5, hold=self.hold)], states)
        camera.start()
        self.assertIsNotNone(camera.latest(7, timeout=2.0), "frames after reconnect expected")
        self.assertEqual(states, [STARTING, CAMERA_LOST, CAMERA_LOST])
        self.assertEqual(camera.state, RUNNING)
        self.assertTrue(first.released and broken.released)

    def test_capture_error_is_handled_like_a_lost_camera(self):
        class CrashingCapture(FakeCapture):
            def read(self):
                raise RuntimeError("decoder crashed")   # e.g. cv2.error

        crashing = CrashingCapture()
        camera = self.make_camera([crashing, FakeCapture(frames=5, hold=self.hold)])
        with self.assertLogs("camera", level="ERROR"):
            camera.start()
            self.assertIsNotNone(camera.latest(4, timeout=2.0), "the reader must survive the error")
        self.assertEqual(camera.state, RUNNING)
        self.assertTrue(crashing.released)

    def test_backoff_restarts_after_frames_even_if_the_capture_then_fails(self):
        class FramesThenCrash(FakeCapture):
            def read(self):
                if self.count < self.frames:
                    return super().read()
                raise RuntimeError("decoder crashed")

        # Three failed opens grow the delay 0.05 -> 0.1 -> 0.2 -> 0.4 s.
        captures = [FakeCapture(opened=False) for _ in range(3)]
        captures += [FramesThenCrash(frames=2), FakeCapture(frames=1, hold=self.hold)]
        opened_at = []

        def open_fn(source, open_timeout_ms, read_timeout_ms):
            opened_at.append(time.perf_counter())
            return captures.pop(0) if captures else FakeCapture(opened=False)

        camera = Camera("http://cam/stream.mjpg", reconnect_min_s=0.05, reconnect_max_s=0.4,
                        open_fn=open_fn)
        self.addCleanup(camera.stop)
        with self.assertLogs("camera", level="ERROR"):
            camera.start()
            self.assertIsNotNone(camera.latest(2, timeout=3.0), "frame after reconnect expected")
        self.assertLess(opened_at[4] - opened_at[3], 0.25,
                        "frames arrived, so the next retry must use the shortest delay")

    def test_credentials_are_redacted(self):
        self.assertEqual(redact("rtsp://admin:secret@10.0.0.5:554/sub"), "rtsp://***@10.0.0.5:554/sub")
        self.assertEqual(redact("http://host:8090/stream.mjpg"), "http://host:8090/stream.mjpg")
        self.assertEqual(redact(0), "0")


if __name__ == "__main__":
    unittest.main()
