"""
stream.py — show what the model sees, as MJPEG over HTTP
========================================================

WHY THIS FILE EXISTS:
    The operator needs to see that the camera finds them: the camera picture
    with the 17 body points and the skeleton drawn on it, plus the last
    gesture.  FT Optix shows it in a WebBrowser object (Chromium), which
    plays MJPEG natively: a never-ending HTTP response where each part is
    one JPEG picture ("multipart/x-mixed-replace").

    http://<device>:8081/             a page that fills the WebBrowser object
                                      and reconnects by itself
    http://<device>:8081/stream.mjpg  the bare stream

COST:
    Each picture is drawn and JPEG-encoded ONCE, in the main loop, and shared
    by every viewer; with no viewer, nothing is drawn or encoded at all.
"""

import logging
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_WIDTH = 640     # bigger camera pictures are scaled down: enough for an HMI tile
# MoveNet keypoint pairs to join: face, arms, torso, legs.
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
GREEN, YELLOW, WHITE = (80, 220, 80), (0, 220, 255), (255, 255, 255)

PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Gesture camera</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#000}img{display:block;width:100%;height:100%;object-fit:contain}</style>
</head><body><img id="v" src="stream.mjpg" alt="">
<script>
var v = document.getElementById("v");
v.onerror = function () { setTimeout(function () { v.src = "stream.mjpg?" + Date.now(); }, 2000); };
</script></body></html>
"""


def _text(img, text, org, scale, color):
    """Text with a dark outline, readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if scale < 1 else 2, cv2.LINE_AA)


def annotate(frame, keypoints, min_score, mirror, gesture=None, info=""):
    """Copy of the frame (at most MAX_WIDTH wide) with the body drawn on it.

    keypoints: (17, 3) of (y, x, score), x and y 0..1 of the padded square
    that pose.py gave the model (the whole frame, centred in the square).
    """
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_WIDTH / w)
    img = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if mirror:
        # WHY: the operator faces the camera; a mirror image moves the way they move.
        img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    side = max(h, w)
    left, top = (side - w) / 2, (side - h) / 2

    points = []
    for y, x, score in keypoints:
        x = 1.0 - x if mirror else x
        points.append((int(x * side - left), int(y * side - top)) if score >= min_score else None)
    for a, b in SKELETON:
        if points[a] and points[b]:
            cv2.line(img, points[a], points[b], GREEN, 2, cv2.LINE_AA)
    for p in points:
        if p:
            cv2.circle(img, p, 4, YELLOW, -1, cv2.LINE_AA)

    if info:
        _text(img, info, (8, 20), 0.5, WHITE)
    if gesture:
        _text(img, gesture.replace("_", " ").upper(), (8, h - 14), 1.0, YELLOW)
    return img


def placeholder(message, width=MAX_WIDTH, height=360):
    """A dark picture with a message, sent while there is no camera picture."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _text(img, message, (20, height // 2), 0.8, WHITE)
    return img


class MjpegServer:
    """Serves the newest annotated picture to every connected viewer."""

    def __init__(self, port, jpeg_quality=70):
        self._params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        self._cond = threading.Condition()
        self._jpeg, self._seq = None, 0
        self.viewers = 0
        self._server = _Server(("0.0.0.0", port), _Handler)
        self._server.stream = self

    def start(self):
        threading.Thread(target=self._server.serve_forever, name="mjpeg", daemon=True).start()
        logger.info("MJPEG stream on http://0.0.0.0:%d/ (stream: /stream.mjpg)", self._server.server_port)

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    def put(self, img):
        """Encode one picture and wake every viewer."""
        ok, jpeg = cv2.imencode(".jpg", img, self._params)
        if ok:
            with self._cond:
                self._jpeg, self._seq = jpeg.tobytes(), self._seq + 1
                self._cond.notify_all()

    def next_jpeg(self, after_seq, timeout):
        """(jpeg, seq) newer than after_seq, or None after timeout s."""
        with self._cond:
            if self._cond.wait_for(lambda: self._seq > after_seq, timeout):
                return self._jpeg, self._seq
        return None

    def _viewer(self, delta):
        with self._cond:
            self.viewers += delta


class _Server(ThreadingHTTPServer):

    def server_bind(self):
        # WHY: HTTPServer.server_bind() looks up its own DNS name, which
        # stalls for seconds without a reachable DNS server (the OptixEdge has
        # none) and is never used here.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "gesture", self.server_address[1]


class _Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif path == "/stream.mjpg":
            self._stream(self.server.stream)
        else:
            self.send_error(404)

    def _stream(self, stream):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        stream._viewer(+1)
        seq = 0
        try:
            while True:
                item = stream.next_jpeg(seq, timeout=5.0)
                if item is None:
                    continue        # main loop busy (start-up): keep the viewer connected
                jpeg, seq = item
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                self.wfile.write(jpeg + b"\r\n")
        except OSError:
            pass                    # viewer closed the page
        finally:
            stream._viewer(-1)

    def log_message(self, format, *args):
        pass                        # no log line per request
