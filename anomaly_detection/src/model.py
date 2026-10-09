import ctypes
import logging
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger("anomaly-detector")


def build_keras_model(input_size: int = 10, hidden_size: int = 64, bottleneck_size: int = 16):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_size,), name="sensor_input")
    encoded = tf.keras.layers.Dense(hidden_size, activation="relu")(inputs)
    encoded = tf.keras.layers.Dense(bottleneck_size, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(hidden_size, activation="relu")(encoded)
    outputs = tf.keras.layers.Dense(input_size, activation="sigmoid")(decoded)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")


def convert_to_tflite(keras_model, tflite_path: str) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    with open(tflite_path, "wb") as f:
        f.write(converter.convert())


def convert_to_tflite_int8(keras_model, tflite_path: str, representative_data, per_tensor: bool = True) -> None:
    """Full-integer post-training quantization: weights, activations, input and output are int8.

    WHY: the i.MX 8M Plus NPU is an integer engine. A graph that still has
    float tensors is left (partly) on the CPU by the VX delegate.
    The representative data must look like production input (scaled window
    means); the converter uses it to choose each tensor's scale and zero point.
    WHY per_tensor: TFLite gives Dense weights one scale per output neuron
    (per-channel) by default; the VIP8000 NPU is documented to prefer one
    scale per tensor.
    """
    import tensorflow as tf

    def representative_dataset():
        for row in np.asarray(representative_data, dtype=np.float32):
            yield [row.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    if per_tensor:
        # Private flag (no public equivalent yet); training/quantize.py prints the
        # resulting kind of every weight tensor, so a silent change is visible.
        converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    with open(tflite_path, "wb") as f:
        f.write(converter.convert())


# ---------------------------------------------------------------------------
# Runtime: everything below runs inside the detector container.
# ---------------------------------------------------------------------------

DEFAULT_VX_DELEGATE = "/usr/lib/libvx_delegate.so"
TRUE_WORDS = ("1", "true", "yes", "on")


def _package_version(distribution: str, module) -> str:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return getattr(module, "__version__", "?")


def _import_runtime():
    """Return (Interpreter, load_delegate, runtime label).

    WHY three options, in this order:
      tflite_runtime  the runtime NXP builds next to its VX delegate; a
                      delegate plugs into the runtime's internals, so this is
                      the pairing NXP tests
      ai_edge_litert  LiteRT, Google's current pip runtime (laptop, CPU)
      tensorflow      a training machine with only full TensorFlow
    """
    try:
        import tflite_runtime
        from tflite_runtime.interpreter import Interpreter, load_delegate

        return Interpreter, load_delegate, f"tflite_runtime {getattr(tflite_runtime, '__version__', '?')}"
    except ImportError:
        pass
    try:
        import ai_edge_litert
        from ai_edge_litert.interpreter import Interpreter, load_delegate

        return Interpreter, load_delegate, f"ai_edge_litert {_package_version('ai-edge-litert', ai_edge_litert)}"
    except ImportError:
        pass
    import tensorflow as tf

    return tf.lite.Interpreter, tf.lite.experimental.load_delegate, f"tensorflow {tf.__version__}"


def resolve_model_settings(model_cfg: dict, environ=None) -> dict:
    """Pick the model file, threshold and accelerator from config.yaml plus env overrides.

    precision "auto" means: int8 when the NPU is requested (the NPU computes
    in integers), float32 otherwise (the original, most accurate model).
    If the NPU later fails, the SAME int8 model runs on the CPU, so the
    threshold still matches the model.
    """
    environ = os.environ if environ is None else environ
    use_npu = bool(model_cfg.get("use_npu", False))
    if "USE_NPU" in environ:
        use_npu = environ["USE_NPU"].strip().lower() in TRUE_WORDS
    precision = environ.get("MODEL_PRECISION", model_cfg.get("precision", "auto")).strip().lower()
    if precision == "auto":
        precision = "int8" if use_npu else "float32"
    if precision not in model_cfg.get("variants", {}):
        raise ValueError(f"model precision '{precision}' has no entry under model.variants in config.yaml")
    variant = model_cfg["variants"][precision]
    return {
        "use_npu": use_npu,
        "precision": precision,
        "model_path": variant["model_path"],
        "threshold_path": variant["threshold_path"],
        "vx_delegate_path": environ.get("VX_DELEGATE_PATH", model_cfg.get("vx_delegate_path", DEFAULT_VX_DELEGATE)),
    }


@contextmanager
def capture_native_output(lines: list):
    """Collect what C/C++ code writes to stdout and stderr (fds 1, 2) inside the block.

    WHY: TFLite and the VX delegate report what they did from C++, below
    Python's sys.stdout/sys.stderr.  TFLite writes to stderr ("Created
    TensorFlow Lite XNNPACK delegate"); the device's VX delegate writes its
    "Vx delegate: ..." lines to STDOUT (measured on the OptixEdge).  Those
    lines are evidence, so we keep them and echo them to the real stderr for
    the container log.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        for fd in (1, 2):
            os.dup2(tmp.fileno(), fd)
        try:
            yield lines
        finally:
            _flush_c_stdio()
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, copy in saved.items():
                os.dup2(copy, fd)
                os.close(copy)
            tmp.seek(0)
            text = tmp.read().decode("utf-8", errors="replace")
            sys.stderr.write(text)
            lines.extend(line.strip() for line in text.splitlines() if line.strip())


def _flush_c_stdio() -> None:
    """fflush(NULL): push C's stdout buffer into the file before we restore fd 1.

    WHY: when stdout is not a terminal, C buffers it; without this the
    delegate's lines would be written later, outside the capture.
    """
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError, TypeError):
        pass  # Windows/laptop: no global C runtime to flush; nothing to lose


_REPLACING = re.compile(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]+)\)")


def delegation_summary(log_lines: list) -> list:
    """Parse TFLite's "Replacing N out of M node(s) with delegate (X)" lines."""
    found = []
    for line in log_lines:
        match = _REPLACING.search(line)
        if match:
            found.append({"delegate": match.group(3), "nodes": int(match.group(1)), "of": int(match.group(2))})
    return found


def graph_delegation(ops: list, delegate_name: str) -> list:
    """Count the model nodes each DELEGATE node replaced, from the graph itself.

    WHY: some TFLite builds (the device's 2.16.2) do not print the
    "Replacing N out of M" line.  The graph still tells the truth: after a
    delegate is applied, the interpreter lists the original ops AND one
    "DELEGATE" node per partition.  An original op belongs to a partition
    when it lies on a path from the partition's inputs to its outputs
    (forward-reachable AND backward-reachable).
    `ops` is interpreter._get_ops_details(): dicts with op_name/inputs/outputs.
    """
    model_ops = [op for op in ops if op["op_name"] != "DELEGATE"]
    found = []
    for node in (op for op in ops if op["op_name"] == "DELEGATE"):
        forward = _reachable(model_ops, set(node["inputs"]), "inputs", "outputs")
        backward = _reachable(model_ops, set(node["outputs"]), "outputs", "inputs")
        found.append({"delegate": delegate_name, "nodes": len(forward & backward), "of": len(model_ops)})
    return found


def _reachable(model_ops: list, tensors: set, take: str, give: str) -> set:
    reached = set()
    changed = True
    while changed:
        changed = False
        for i, op in enumerate(model_ops):
            if i not in reached and tensors.intersection(op[take]):
                reached.add(i)
                tensors.update(t for t in op[give] if t >= 0)
                changed = True
    return reached


class InferenceEngine:
    """One TFLite interpreter plus the facts that prove where it runs.

    accelerator   "npu" only after the VX delegate loaded AND a first
                  inference succeeded; otherwise "cpu"
    warmup_ms     first inference; on the NPU it includes compiling the
                  graph for the hardware, so it is much longer than later runs
    inference_ms  duration of the most recent reconstruct() call
    """

    def __init__(self, model_path: str, use_npu: bool = False,
                 vx_delegate_path: str = DEFAULT_VX_DELEGATE, num_threads: int = 1):
        Interpreter, load_delegate, self.runtime = _import_runtime()
        self.model_path = model_path
        self.vx_delegate_path = vx_delegate_path
        self.npu_requested = use_npu
        self.accelerator = "cpu"
        self.npu_error = None
        self.native_log = []
        self.interpreter = None
        self.inference_ms = None
        self._lock = threading.Lock()

        with capture_native_output(self.native_log):
            if use_npu:
                try:
                    delegate = load_delegate(vx_delegate_path)
                    self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads,
                                                   experimental_delegates=[delegate])
                    self.interpreter.allocate_tensors()
                    # WHY inside the try: the NPU compiles the graph at the first
                    # invoke, which is where a mismatched driver fails.
                    self.warmup_ms = self._timed_zero_run()
                    self.accelerator = "npu"
                except Exception as exc:
                    # WHY broad: a missing or mismatched NPU stack fails as OSError,
                    # ValueError or RuntimeError depending on the stage; the
                    # detector must keep working on the CPU in every case.
                    self.npu_error = f"{type(exc).__name__}: {exc}"
                    self.interpreter = None
            if self.interpreter is None:
                self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
                self.interpreter.allocate_tensors()
                self.warmup_ms = self._timed_zero_run()

        if self.npu_error:
            logger.warning("NPU requested but unusable (%s); running on CPU", self.npu_error)
        self.precision = np.dtype(self.interpreter.get_input_details()[0]["dtype"]).name
        logger.info("Model %s (%s) on %s via %s, warm-up %.1f ms", os.path.basename(model_path),
                    self.precision, self.accelerator.upper(), self.runtime, self.warmup_ms)

    def _timed_zero_run(self) -> float:
        detail = self.interpreter.get_input_details()[0]
        self.interpreter.set_tensor(detail["index"], np.zeros(detail["shape"], dtype=detail["dtype"]))
        start = time.perf_counter()
        self.interpreter.invoke()
        return (time.perf_counter() - start) * 1000.0

    def reconstruct(self, feature_vector) -> np.ndarray:
        from detector import reconstruct

        # WHY a lock: a TFLite interpreter is not thread-safe, and the
        # benchmark thread shares it with the sensor handler.
        with self._lock:
            start = time.perf_counter()
            output = reconstruct(self.interpreter, feature_vector)
            self.inference_ms = (time.perf_counter() - start) * 1000.0
        return output

    def describe(self) -> dict:
        return {
            "accelerator": self.accelerator,
            "npu_requested": self.npu_requested,
            "npu_error": self.npu_error,
            "model": os.path.basename(self.model_path),
            "precision": self.precision,
            "runtime": self.runtime,
            "delegate": self.vx_delegate_path if self.accelerator == "npu" else None,
            "delegation": self.delegation(),
            "warmup_ms": round(self.warmup_ms, 3),
        }

    def delegation(self) -> list:
        """Which delegate took how many nodes: from the log if it says so, else from the graph."""
        from_log = delegation_summary(self.native_log)
        if from_log:
            return from_log
        try:
            ops = self.interpreter._get_ops_details()
        except Exception:  # private API: absent in some runtimes, evidence is optional
            return []
        name = "Vx Delegate" if self.accelerator == "npu" else "XNNPACK"
        return graph_delegation(ops, name)
