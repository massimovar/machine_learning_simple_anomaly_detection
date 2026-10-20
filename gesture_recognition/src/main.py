"""
main.py — gesture recognition service
=====================================

WHY THIS FILE EXISTS:
    Wires the pieces together and runs the loop

        camera (newest frame) -> MoveNet (17 keypoints) -> gesture rules -> MQTT
                                                       +-> MJPEG picture with the skeleton

    and reports health three ways:
      - MQTT status every few seconds and at once on camera changes (FT Optix shows it)
      - a heartbeat file for the container HEALTHCHECK
      - a watchdog that ends the process if the loop hangs, so the container
        restart policy starts a fresh one

RUN:
    container:  podman compose -f gesture_recognition/docker-compose.yml up --build
    laptop:     set CAMERA_URL=0 and MQTT_HOST=localhost, then
                python gesture_recognition/src/main.py
"""

import logging
import os
import signal
import sys
import tempfile
import threading
import time
from collections import deque

import yaml

from camera import CAMERA_LOST, RUNNING, Camera
from gestures import GestureConfig, GestureEngine
from mqtt_client import GesturePublisher, make_status
from pose import PoseEstimator
from stream import MjpegServer, annotate, placeholder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("gesture-recognizer")

# WHY: paths in config.yaml are relative to the gesture_recognition folder,
# so the service runs the same from any working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEARTBEAT_FILE = os.path.join(tempfile.gettempdir(), "gesture_heartbeat")
WATCHDOG_S = 15
# WHY so long: the first NPU inference compiles the graph (no measured
# duration on the i.MX 8M Plus yet); a hang there still ends in a restart.
STARTUP_S = 120
LOG_EVERY_N_STATUS = 15
SHOW_GESTURE_S = 2.0   # the MJPEG picture names a recognised gesture this long


def load_config(path):
    """Read config.yaml; non-empty environment variables override the
    values that differ between laptop and edge device."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env = os.environ
    if env.get("CAMERA_URL"):
        config["camera"]["url"] = env["CAMERA_URL"]
    if env.get("MQTT_HOST"):
        config["mqtt"]["broker_host"] = env["MQTT_HOST"]
    if env.get("MQTT_PORT"):
        config["mqtt"]["broker_port"] = int(env["MQTT_PORT"])
    if env.get("USE_NPU"):
        config["model"]["use_npu"] = env["USE_NPU"].strip().lower() in ("1", "true", "yes")
    return config


class Watchdog(threading.Thread):
    """Exit the process if the main loop stops making progress.

    WHY: container restart policies restart containers that EXIT; a hung
    process (stuck driver, stuck stream) would otherwise stay hung, even
    when marked unhealthy.
    """

    def __init__(self, timeout_s, startup_s):
        super().__init__(name="watchdog", daemon=True)
        self._timeout_s = timeout_s
        self._deadline = time.monotonic() + startup_s

    def feed(self):
        self._deadline = time.monotonic() + self._timeout_s

    def run(self):
        while True:
            time.sleep(1)
            if time.monotonic() > self._deadline:
                logger.critical("No progress before the watchdog deadline; exiting so the container restarts")
                os._exit(1)


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main():
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    config = load_config(os.path.join(ROOT, "config.yaml"))
    cam_cfg, model_cfg, mqtt_cfg = config["camera"], config["model"], config["mqtt"]
    engine = GestureEngine(GestureConfig(**config["gestures"]), mirror=cam_cfg["mirror"])
    min_score = config["gestures"]["min_keypoint_score"]

    stream = MjpegServer(config["stream"]["port"], config["stream"]["jpeg_quality"])
    stream.start()
    publisher = GesturePublisher(mqtt_cfg["broker_host"], mqtt_cfg["broker_port"], mqtt_cfg["client_id"],
                                 mqtt_cfg["events_topic"], mqtt_cfg["status_topic"])
    publisher.start()
    watchdog = Watchdog(WATCHDOG_S, STARTUP_S)
    watchdog.start()
    estimator = PoseEstimator(os.path.join(ROOT, model_cfg["model_path"]), model_cfg["use_npu"],
                              model_cfg["vx_delegate_path"], model_cfg["num_threads"])
    accelerator = estimator.accelerator
    publisher.publish_status(make_status("starting", accelerator=accelerator))
    estimator.warm_up()

    camera = Camera(cam_cfg["url"], cam_cfg["open_timeout_ms"], cam_cfg["read_timeout_ms"],
                    reconnect_max_s=cam_cfg["reconnect_max_s"])
    camera.start()

    period = 1.0 / model_cfg["target_fps"]
    window_s = mqtt_cfg["status_interval_s"]
    seq, state = 0, None
    fps, shown_gesture, shown_until = 0.0, None, 0.0
    status_count, next_status = 0, time.perf_counter()
    recent = deque()   # (t_done, inference_ms, processing_ms) of frames in the last window_s
    try:
        while not stop.is_set():
            now = time.perf_counter()
            if camera.state != state:
                state = camera.state
                if state == CAMERA_LOST:
                    engine.reset()
                next_status = now          # report camera changes at once

            if now >= next_status:
                while recent and recent[0][0] < now - window_s:
                    recent.popleft()
                fps = len(recent) / window_s
                inference_p50 = percentile([r[1] for r in recent], 0.5)
                status = make_status(state, engine.person_present, fps, inference_p50, accelerator)
                publisher.publish_status(status)
                with open(HEARTBEAT_FILE, "w") as f:
                    f.write(status["timestamp"])
                if status_count % LOG_EVERY_N_STATUS == 0:
                    logger.info("state=%s fps=%.1f inference p50=%.1f ms processing p95=%.1f ms "
                                "person=%s accelerator=%s", state, fps, inference_p50,
                                percentile([r[2] for r in recent], 0.95), engine.person_present, accelerator)
                status_count += 1
                next_status = now + window_s

            item = camera.latest(seq, timeout=0.5)
            if item is None:
                # WHY: waiting is normal while the camera (re)connects, but a
                # "running" camera that delivers nothing is stuck: no feed, so
                # the watchdog restarts the service.
                if camera.state != RUNNING:
                    watchdog.feed()
                    if stream.viewers:
                        stream.put(placeholder(f"Camera: {camera.state}"))
                continue
            watchdog.feed()
            frame, t_frame, seq = item
            t0 = time.perf_counter()
            keypoints = estimator.infer(frame)
            t1 = time.perf_counter()
            event = engine.update(t_frame, keypoints)
            t2 = time.perf_counter()
            recent.append((t2, (t1 - t0) * 1000, (t2 - t_frame) * 1000))   # processing: frame received -> decision
            if event is not None:
                publisher.publish_event(event.gesture, event.confidence)
                shown_gesture, shown_until = event.gesture, t2 + SHOW_GESTURE_S
            if stream.viewers:
                info = f"{accelerator.upper()}  {fps:.0f} fps  {(t1 - t0) * 1000:.0f} ms"
                stream.put(annotate(frame, keypoints, min_score, cam_cfg["mirror"],
                                    shown_gesture if t2 < shown_until else None, info))

            spare = period - (time.perf_counter() - t0)   # rate cap: at most target_fps
            if spare > 0:
                stop.wait(spare)
    finally:
        logger.info("Stopping")
        camera.stop()
        stream.stop()
        publisher.stop(make_status("offline", accelerator=accelerator))


if __name__ == "__main__":
    main()
