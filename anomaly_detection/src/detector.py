"""
detector.py — Anomaly scoring and threshold logic
===================================================

WHY THIS FILE EXISTS:
    The autoencoder outputs a "reconstruction" of its input.  This file answers
    the question: **"Is that reconstruction bad enough to call it an anomaly?"**

    We compute the Mean Squared Error (MSE) between the original and the
    reconstruction.  If MSE > threshold → anomaly.

KEY CONCEPT — Reconstruction Error as Anomaly Score:
    During training, the autoencoder learned what "normal" looks like.
    For normal data, it reconstructs accurately → low MSE.
    For abnormal data, it fails to reconstruct → high MSE.

    The threshold was computed during training as:
        threshold = mean(training_MSEs) + 3 × std(training_MSEs)

    This is the "three-sigma rule": in a normal (Gaussian) distribution,
    99.7 % of values fall within 3 standard deviations of the mean.
    Anything beyond that is extremely unlikely under normal conditions.

INFERENCE BACKEND — TFLite Interpreter:
    Instead of PyTorch, inference is performed through a TFLite Interpreter.
    The interpreter loads the flat .tflite file, sets the input tensor, calls
    invoke(), and reads the output tensor — no PyTorch or autograd overhead.
    On the NXP i.MX 8M Plus the interpreter can be extended with the NPU
    delegate (see model.load_tflite_model) to offload computation to the NPU.
"""

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    """
    Holds the outcome of one anomaly check.

    Attributes
    ----------
    is_anomaly : bool
        True if the anomaly score exceeded the threshold.
    score : float
        The MSE between original and reconstruction (the anomaly score).
    threshold : float
        The decision boundary loaded from training.
    """
    is_anomaly: bool
    score: float
    threshold: float

    def to_alert_json(self) -> str:
        """Convert this result to a JSON string suitable for publishing via MQTT."""
        return json.dumps({
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.score, 6),
            "threshold": round(self.threshold, 6),
        })


def load_threshold(threshold_path: str) -> float:
    """
    Load the anomaly threshold from a JSON file saved during training.

    Returns
    -------
    float
        The threshold value.
    """
    with open(threshold_path, "r") as f:
        data = json.load(f)
    return data["threshold"]


def compute_anomaly_score(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) between the original input and the
    autoencoder's reconstruction.

    Formula:
        MSE = (1/n) × Σ (original_i − reconstructed_i)²

    Parameters
    ----------
    original : np.ndarray
        The normalised feature vector fed into the model, shape (n_tags,).
    reconstructed : np.ndarray
        The model's output, same shape.

    Returns
    -------
    float
        The MSE value (lower = more normal, higher = more anomalous).
    """
    # np.mean computes the average; (a - b)**2 is element-wise squared difference
    return float(np.mean((original - reconstructed) ** 2))


def detect(interpreter, feature_vector: np.ndarray, threshold: float) -> DetectionResult:
    """
    Run the full detection pipeline on one preprocessed feature vector.

    Steps:
    1. Write the feature vector into the TFLite interpreter's input tensor.
    2. Call interpreter.invoke() to run the model.
    3. Read the output tensor (the reconstruction).
    4. Compute MSE between input and reconstruction.
    5. Compare MSE to threshold.

    Parameters
    ----------
    interpreter : tflite.Interpreter
        An allocated TFLite Interpreter returned by model.load_tflite_model().
        On the NXP i.MX 8M Plus this may include the NPU delegate for
        hardware-accelerated inference.
    feature_vector : np.ndarray
        Normalised, shape (n_tags,), values in [0, 1].
    threshold : float
        Decision boundary from training.

    Returns
    -------
    DetectionResult
    """
    # --- Step 1: get tensor handles -----------------------------------------------
    # TFLite models have named input/output tensors; we query their indices once.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # --- Step 2: write input (shape must match: 1 × n_tags) -----------------------
    # TFLite expects float32 and a batch dimension even for single samples.
    input_data = feature_vector.astype(np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # --- Step 3: run inference ---------------------------------------------------
    # invoke() executes the full graph (encoder + decoder).
    interpreter.invoke()

    # --- Step 4: read output and compute MSE ------------------------------------
    reconstructed = interpreter.get_tensor(output_details[0]["index"])[0]  # remove batch dim
    score = compute_anomaly_score(feature_vector, reconstructed)

    # --- Step 5: compare to threshold -------------------------------------------
    is_anomaly = score > threshold

    return DetectionResult(is_anomaly=is_anomaly, score=score, threshold=threshold)
