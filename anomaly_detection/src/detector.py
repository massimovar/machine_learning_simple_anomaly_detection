import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    is_anomaly: bool
    score: float
    threshold: float

    def to_alert_json(self, extra: dict = None) -> str:
        """extra: evidence fields (accelerator, precision, inference_ms) appended after the core three."""
        return json.dumps(
            {
                "is_anomaly": self.is_anomaly,
                "anomaly_score": round(self.score, 6),
                "threshold": round(self.threshold, 6),
                **(extra or {}),
            }
        )


def load_threshold(threshold_path: str) -> float:
    with open(threshold_path, "r", encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def compute_anomaly_score(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original - reconstructed) ** 2))


def quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    """float -> the tensor's dtype. A float tensor passes through unchanged.

    int8 stores a real number r as q = round(r / scale) + zero_point, clipped
    to [-128, 127]. Values outside the calibrated range saturate at the edges.
    """
    dtype = detail["dtype"]
    values = np.asarray(values, dtype=np.float32)
    if dtype == np.float32:
        return values
    scale, zero_point = detail["quantization"]
    info = np.iinfo(dtype)
    q = np.round(values / scale) + zero_point
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    """The tensor's dtype -> float: r = (q - zero_point) * scale."""
    if detail["dtype"] == np.float32:
        return np.asarray(values, dtype=np.float32)
    scale, zero_point = detail["quantization"]
    return ((values.astype(np.float32) - zero_point) * scale).astype(np.float32)


def reconstruct(interpreter, feature_vector: np.ndarray) -> np.ndarray:
    """Run the autoencoder once; works for float32 and int8 models alike."""
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_data = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)

    interpreter.set_tensor(input_detail["index"], quantize(input_data, input_detail))
    interpreter.invoke()
    return dequantize(interpreter.get_tensor(output_detail["index"]), output_detail)[0]


def detect(engine, feature_vector: np.ndarray, threshold: float) -> DetectionResult:
    """engine: anything with reconstruct(vector) -> vector, e.g. model.InferenceEngine."""
    original = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
    reconstructed = engine.reconstruct(original)
    # WHY compare against the float input, not the quantized one: an anomaly
    # can push a value outside the int8 range, where it saturates. The model
    # then sees a clipped value, but the score still measures the real gap.
    score = compute_anomaly_score(original, reconstructed)
    return DetectionResult(score > threshold, score, threshold)
