"""Tests for stream.py — where the body points land, and a real MJPEG round trip."""

import os
import sys
import threading
import time
import unittest
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stream import YELLOW, MjpegServer, annotate  # noqa: E402


def one_point(y, x, score=1.0):
    """Keypoints with only the nose set."""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = (y, x, score)
    return kps


def is_yellow(img, x, y):
    return tuple(int(c) for c in img[y, x]) == YELLOW


class TestAnnotate(unittest.TestCase):

    def test_point_maps_from_the_padded_square(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        img = annotate(frame, one_point(0.5, 0.25), 0.3, mirror=False)
        # square: 640 px, frame starts 80 px below its top edge
        self.assertTrue(is_yellow(img, 160, 240))

    def test_mirror_flips_the_point(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        img = annotate(frame, one_point(0.5, 0.25), 0.3, mirror=True)
        self.assertTrue(is_yellow(img, 480, 240))

    def test_big_frame_is_scaled_to_640(self):
        frame = np.zeros((1296, 2304, 3), dtype=np.uint8)
        img = annotate(frame, one_point(0.5, 0.5), 0.3, mirror=False)
        self.assertEqual(img.shape, (360, 640, 3))
        self.assertTrue(is_yellow(img, 320, 180))

    def test_unsure_points_are_not_drawn(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        img = annotate(frame, one_point(0.5, 0.5, score=0.1), 0.3, mirror=False)
        self.assertFalse(is_yellow(img, 320, 240))

    def test_camera_frame_is_not_modified(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotate(frame, one_point(0.5, 0.5), 0.3, mirror=False, gesture="swipe_left", info="CPU")
        self.assertEqual(frame.max(), 0)


class TestMjpegServer(unittest.TestCase):

    def setUp(self):
        self.server = MjpegServer(port=0)
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server._server.server_port}"

    def tearDown(self):
        self.server.stop()

    def test_page_embeds_the_stream(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as r:
            self.assertIn(b'src="stream.mjpg"', r.read())

    def test_stream_delivers_jpeg_parts_and_counts_viewers(self):
        self.assertEqual(self.server.viewers, 0)

        def feed():
            while self.server.viewers == 0:
                time.sleep(0.01)
            self.server.put(np.zeros((360, 640, 3), dtype=np.uint8))

        threading.Thread(target=feed, daemon=True).start()
        with urllib.request.urlopen(self.base + "/stream.mjpg", timeout=5) as r:
            self.assertTrue(r.headers["Content-Type"].startswith("multipart/x-mixed-replace"))
            head = r.read(200)
        self.assertIn(b"Content-Type: image/jpeg", head)
        self.assertIn(b"\xff\xd8", head)          # JPEG start-of-image marker

    def test_unknown_path_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/nope", timeout=5)
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
