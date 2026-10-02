"""
webcam_bridge.py — share the laptop webcam as an MJPEG stream (development only)
================================================================================

WHY THIS FILE EXISTS:
    On Windows, Podman runs containers inside a WSL2 virtual machine that
    cannot open the laptop's USB webcam.  This script runs on the Windows
    host, reads the webcam and serves it as an MJPEG-over-HTTP stream: the
    same kind of input an Ethernet camera gives in production, so the
    container code is identical in both environments.

    Frames are streamed only, never saved.

RUN (on the host, not in a container; needs opencv-python-headless):
    python gesture_recognition/tools/webcam_bridge.py --bind <address>
    check the picture: open the URL it prints in a browser

SECURITY:
    Binds to 127.0.0.1 by default, so the webcam is not reachable from the
    network.  Containers in the Podman (WSL) machine on Windows cannot reach
    127.0.0.1: bind to the WSL adapter address instead, which only this PC
    can reach (see the header of docker-compose.yml).
"""

import argparse
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("webcam-bridge")


class LatestJpeg:
    """Newest encoded frame, shared by all viewers (encoded once per frame)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg, self._seq = None, 0

    def put(self, jpeg):
        with self._cond:
            self._jpeg, self._seq = jpeg, self._seq + 1
            self._cond.notify_all()

    def newer_than(self, seq, timeout=5.0):
        """(jpeg, seq) of the next new frame, or None if none arrives in time."""
        with self._cond:
            if self._cond.wait_for(lambda: self._seq > seq, timeout):
                return self._jpeg, self._seq
        return None


class StreamHandler(BaseHTTPRequestHandler):
    frames = None   # LatestJpeg, set in main()

    def do_GET(self):
        if self.path != "/stream.mjpg":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq = 0
        try:
            while True:
                item = self.frames.newer_than(seq)
                if item is None:
                    return          # webcam stalled: closing lets the reader reconnect
                jpeg, seq = item
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                self.wfile.write(jpeg + b"\r\n")
        except ConnectionError:
            pass                    # viewer disconnected

    def log_message(self, format, *args):
        pass                        # no log line per request


def main():
    parser = argparse.ArgumentParser(description="Serve the local webcam as MJPEG over HTTP.")
    parser.add_argument("--device", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    parser.add_argument("--bind", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    # WHY DirectShow on Windows: opens the webcam in about a second, while the
    # default Media Foundation backend can take much longer.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.device, backend)
    if not cap.isOpened():
        sys.exit(f"Cannot open webcam {args.device} (in use by another app?)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    StreamHandler.frames = frames = LatestJpeg()
    server = ThreadingHTTPServer((args.bind, args.port), StreamHandler)
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    logger.info("Streaming webcam %d on http://%s:%d/stream.mjpg (Ctrl+C to stop)",
                args.device, args.bind, args.port)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Webcam stopped delivering frames")
                break
            ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                frames.put(jpeg.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        cap.release()


if __name__ == "__main__":
    main()
