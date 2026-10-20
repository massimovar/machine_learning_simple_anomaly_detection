import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger("anomaly-detector")

DEFAULT_VX_DELEGATE = "/usr/lib/libvx_delegate.so"


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
        from importlib.metadata import version

        from ai_edge_litert.interpreter import Interpreter, load_delegate

        return Interpreter, load_delegate, f"ai_edge_litert {version('ai-edge-litert')}"
    except ImportError:
        pass
    import tensorflow as tf

    return tf.lite.Interpreter, tf.lite.experimental.load_delegate, f"tensorflow {tf.__version__}"


def quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    """float -> the tensor's dtype. A float tensor passes through unchanged.

    int8 stores a real number r as q = round(r / scale) + zero_point, clipped
    to [-128, 127]. Values outside the calibrated range saturate at the edges.
    """
    values = np.asarray(values, dtype=np.float32)
    if detail["dtype"] == np.float32:
        return values
    scale, zero_point = detail["quantization"]
    info = np.iinfo(detail["dtype"])
    return np.clip(np.round(values / scale) + zero_point, info.min, info.max).astype(detail["dtype"])


def dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    """The tensor's dtype -> float: r = (q - zero_point) * scale."""
    if detail["dtype"] == np.float32:
        return np.asarray(values, dtype=np.float32)
    scale, zero_point = detail["quantization"]
    return ((values.astype(np.float32) - zero_point) * scale).astype(np.float32)


def graph_delegation(ops: list, delegate_name: str) -> list:
    """Count the model nodes each DELEGATE node replaced, from the graph itself.

    WHY the graph and not TFLite's "Replacing N out of M node(s)" log line:
    some TFLite builds (the device's 2.16.2) do not print it.  After a
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

    def __init__(self, model_path: str, use_npu: bool = False, vx_delegate_path: str = DEFAULT_VX_DELEGATE):
        Interpreter, load_delegate, self.runtime = _import_runtime()
        self.model_path = model_path
        self.vx_delegate_path = vx_delegate_path
        self.npu_requested = use_npu
        self.accelerator = "cpu"
        self.npu_error = None
        self.interpreter = None
        self.inference_ms = None
        self._lock = threading.Lock()

        if use_npu:
            try:
                delegate = load_delegate(vx_delegate_path)
                self.interpreter = Interpreter(model_path=model_path, num_threads=1,
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
                logger.warning("NPU requested but unusable (%s); running on CPU", self.npu_error)
        if self.interpreter is None:
            self.interpreter = Interpreter(model_path=model_path, num_threads=1)
            self.interpreter.allocate_tensors()
            self.warmup_ms = self._timed_zero_run()

        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]
        self.precision = np.dtype(self._input["dtype"]).name
        logger.info("Model %s (%s) on %s via %s, warm-up %.1f ms", os.path.basename(model_path),
                    self.precision, self.accelerator.upper(), self.runtime, self.warmup_ms)

    def _timed_zero_run(self) -> float:
        detail = self.interpreter.get_input_details()[0]
        self.interpreter.set_tensor(detail["index"], np.zeros(detail["shape"], dtype=detail["dtype"]))
        start = time.perf_counter()
        self.interpreter.invoke()
        return (time.perf_counter() - start) * 1000.0

    def reconstruct(self, feature_vector) -> np.ndarray:
        """Run the autoencoder once; works for float32 and int8 models alike."""
        input_data = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        # WHY a lock: a TFLite interpreter is not thread-safe, and the
        # benchmark thread shares it with the sensor handler.
        with self._lock:
            start = time.perf_counter()
            self.interpreter.set_tensor(self._input["index"], quantize(input_data, self._input))
            self.interpreter.invoke()
            output = dequantize(self.interpreter.get_tensor(self._output["index"]), self._output)[0]
            self.inference_ms = (time.perf_counter() - start) * 1000.0
        return output

    def score(self, feature_vector) -> float:
        """Anomaly score: mean squared error between the input and its reconstruction.

        WHY compare against the float input, not the quantized one: an anomaly
        can push a value outside the int8 range, where it saturates. The model
        then sees a clipped value, but the score still measures the real gap.
        """
        original = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
        return float(np.mean((original - self.reconstruct(original)) ** 2))

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
        """Which delegate took how many nodes, read from the delegated graph."""
        try:
            ops = self.interpreter._get_ops_details()
        except Exception:  # private API: absent in some runtimes, evidence is optional
            return []
        return graph_delegation(ops, "Vx Delegate" if self.accelerator == "npu" else "XNNPACK")
