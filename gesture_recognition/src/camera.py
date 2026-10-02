"""
camera.py — always hand out the newest camera frame
====================================================

WHY THIS FILE EXISTS:
    Network cameras deliver frames at their own pace.  If frames were read
    only when the model is ready, old frames would pile up in buffers and
    the reaction to a gesture would get later and later.  A background
    thread therefore reads continuously and keeps ONLY the newest frame
    (a "latest-frame slot"); older frames are simply dropped.

    The same thread handles failures: when no frame arrives within the read
    timeout the camera is "lost", the stream is closed and re-opened after a
    growing pause (1, 2, 4, 8, 10 s ...), so unplugging the camera at a
    trade show never needs a restart.
"""

import logging
import re
import threading
import time

import cv2

logger = logging.getLogger(__name__)

# Camera states; the service publishes them unchanged as MQTT status "state".
STARTING, RUNNING, CAMERA_LOST = "starting", "running", "camera_lost"


def redact(source):
    """Hide user:password in camera URLs before they reach the logs."""
    return re.sub(r"//[^/@]+@", "//***@", str(source))


def open_capture(source, open_timeout_ms, read_timeout_ms):
    """Open a webcam index ("0"), a network stream or a video file."""
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms,
    ])


class Camera:
    """Background reader with a latest-frame slot and automatic reconnect."""

    def __init__(self, source, open_timeout_ms=5000, read_timeout_ms=2000,
                 reconnect_min_s=1.0, reconnect_max_s=10.0, open_fn=open_capture):
        self.source = source
        self.state = STARTING
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._reconnect_min_s = reconnect_min_s
        self._reconnect_max_s = reconnect_max_s
        self._open_fn = open_fn
        # WHY: replay a video file at its own frame rate so gesture timing
        # stays real; live sources are never throttled.
        self._is_file = "://" not in str(source) and not str(source).isdigit()
        self._cond = threading.Condition()
        # WHY perf_counter: frame times feed speed and latency math, and
        # time.monotonic() only ticks every ~16 ms on Windows (Python < 3.13).
        self._latest = None        # (frame, t_perf_counter, seq)
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def latest(self, after_seq, timeout):
        """Newest (frame, t, seq) with seq > after_seq, or None after timeout s."""
        with self._cond:
            if self._cond.wait_for(lambda: self._seq > after_seq, timeout):
                return self._latest
        return None

    def _run(self):
        delay = self._reconnect_min_s
        while not self._stop.is_set():
            seq = self._seq
            try:
                cap = self._open_fn(self.source, self._open_timeout_ms, self._read_timeout_ms)
                try:
                    if cap.isOpened():
                        logger.info("Camera opened: %s", redact(self.source))
                        self._read_frames(cap)
                finally:
                    cap.release()
            except Exception:
                # WHY: an unexpected OpenCV error must not end this thread
                # silently (the service would wait for frames forever); it is
                # handled like a lost camera.
                logger.exception("Camera %s failed", redact(self.source))
            if self._seq != seq:
                # WHY: frames arrived, so the camera works: retry quickly,
                # even if this connection then ended with an error.
                delay = self._reconnect_min_s
            if self._stop.is_set():
                break
            self.state = CAMERA_LOST
            logger.warning("Camera %s lost; retrying in %.0f s", redact(self.source), delay)
            self._stop.wait(delay)
            delay = min(delay * 2, self._reconnect_max_s)

    def _read_frames(self, cap):
        """Read until a failure or stop()."""
        period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if self._is_file else 0.0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            with self._cond:
                self._seq += 1
                self._latest = (frame, time.perf_counter(), self._seq)
                self.state = RUNNING
                self._cond.notify_all()
            if period:
                self._stop.wait(period)
