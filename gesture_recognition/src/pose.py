"""
pose.py — run MoveNet on a camera frame
=======================================

WHY THIS FILE EXISTS:
    Wraps the TFLite interpreter so the rest of the service only sees
    "camera frame in -> 17 body keypoints out".  It also decides where the
    model runs: on the i.MX 8M Plus NPU (through NXP's VX delegate) or on
    the CPU.  If the NPU cannot be used, it falls back to the CPU and says
    so in the log and in the MQTT status ("accelerator").

PIPELINE (cheapest order first):
    1. centre square crop: MoveNet expects a square image, and the middle of
       the frame is where the operator stands.
    2. resize to 192x192: the model's input size.
    3. BGR -> RGB on the small image: OpenCV delivers BGR; converting 192x192
       pixels instead of the full frame saves CPU.
    4. inference -> (17, 3) array of (y, x, score); x and y are 0..1 of the crop.
"""

import logging

import cv2
import numpy as np

# WHY three options:
#   ai_edge_litert  LiteRT, Google's current TFLite runtime (pip, any machine);
#                   much faster on the CPU than the old tflite_runtime 2.14
#                   (laptop container: 3 ms instead of 33 ms per frame)
#   tflite_runtime  the runtime NXP ships in its BSP next to the NPU delegate
#   tensorflow      a laptop that only has full TensorFlow installed
try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        load_delegate = tf.lite.experimental.load_delegate

logger = logging.getLogger(__name__)


def center_square(frame):
    """Largest square in the middle of the frame (a view, no copy)."""
    h, w = frame.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return frame[top:top + side, left:left + side]


def preprocess(frame, width, height):
    """BGR camera frame -> uint8 RGB batch of shape [1, height, width, 3]."""
    small = cv2.resize(center_square(frame), (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)[np.newaxis]


class PoseEstimator:
    def __init__(self, model_path, use_npu=False, vx_delegate_path="/usr/lib/libvx_delegate.so",
                 num_threads=2):
        self.accelerator = "cpu"
        self._interpreter = None
        if use_npu:
            try:
                self._interpreter = Interpreter(model_path=model_path, num_threads=num_threads,
                                                experimental_delegates=[load_delegate(vx_delegate_path)])
                self._interpreter.allocate_tensors()
                # WHY: the NPU compiles the graph at the first inference, which
                # is where a mismatched driver fails; doing it here keeps the
                # CPU fallback below in charge of that failure too.
                self._interpreter.invoke()
                self.accelerator = "npu"
            except Exception as exc:
                # WHY broad: a missing or mismatched NPU stack fails as OSError,
                # ValueError or RuntimeError depending on the stage; the demo
                # must keep working on the CPU in every case.
                logger.warning("NPU delegate %s unusable (%s); falling back to CPU", vx_delegate_path, exc)
                self._interpreter = None
        if self._interpreter is None:
            self._interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
            self._interpreter.allocate_tensors()

        inp = self._interpreter.get_input_details()[0]
        if inp["dtype"] != np.uint8:
            raise ValueError(f"{model_path}: expected uint8 input (MoveNet int8), got {inp['dtype']}")
        _, self.height, self.width, _ = (int(v) for v in inp["shape"])
        self._input_index = inp["index"]
        self._output_index = self._interpreter.get_output_details()[0]["index"]
        logger.info("MoveNet %dx%d loaded on %s", self.width, self.height, self.accelerator.upper())

    def infer(self, frame):
        """BGR frame -> (17, 3) float32 array of (y, x, score)."""
        self._interpreter.set_tensor(self._input_index, preprocess(frame, self.width, self.height))
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output_index)[0, 0]

    def warm_up(self, runs=3):
        """Run a few dummy inferences.

        WHY: the first inferences are slow (NPU graph compilation, CPU
        buffer setup); doing them before the camera starts avoids a frozen
        first gesture.
        """
        dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for _ in range(runs):
            self.infer(dummy)
