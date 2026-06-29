import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    is_anomaly: bool
    score: float
    threshold: float

    def to_alert_json(self) -> str:
        return json.dumps(
            {
                "is_anomaly": self.is_anomaly,
                "anomaly_score": round(self.score, 6),
                "threshold": round(self.threshold, 6),
            }
        )


def load_threshold(threshold_path: str) -> float:
    with open(threshold_path, "r", encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def compute_anomaly_score(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original - reconstructed) ** 2))


def detect(interpreter, feature_vector: np.ndarray, threshold: float) -> DetectionResult:
    input_data = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    reconstructed = interpreter.get_tensor(output_details[0]["index"])[0]
    score = compute_anomaly_score(input_data[0], reconstructed)
    return DetectionResult(score > threshold, score, threshold)
