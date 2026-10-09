"""
quantize.py — create the int8 (NPU-ready) twin of the float autoencoder
=======================================================================

WHY THIS FILE EXISTS:
    The i.MX 8M Plus NPU computes with 8-bit integers. Our autoencoder is
    float32, so the VX delegate cannot put it on the NPU as it is.

    This script does NOT retrain. It reads the weights out of
    models/autoencoder.tflite, rebuilds the identical Keras network, and
    converts that network to full int8. Only one thing changes (how the
    numbers are stored), so any difference between the two models is
    caused by quantization alone.

OUTPUT:
    models/autoencoder_int8.tflite   int8 weights, activations, input, output
    models/threshold_int8.json       threshold calibrated on the int8 model
    a float-vs-int8 comparison printed to the console

RUN (from anomaly_detection/, needs full TensorFlow):
    python training/quantize.py                 # per-tensor weights (NPU-friendly, default)
    python training/quantize.py --per-channel   # TFLite's default, for comparison
"""

import json
import os
import pickle
import sys

import numpy as np
import yaml

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from detector import compute_anomaly_score, reconstruct  # noqa: E402
from model import build_keras_model, convert_to_tflite_int8  # noqa: E402
from train import compute_threshold, load_csv, simulate_windowing  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")
INT8_MODEL = os.path.join(MODELS_DIR, "autoencoder_int8.tflite")
INT8_THRESHOLD = os.path.join(MODELS_DIR, "threshold_int8.json")


def _interpreter(path: str):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def keras_from_tflite(tflite_path: str, model_cfg: dict):
    """Rebuild the Keras autoencoder with the exact weights stored in a float .tflite."""
    interpreter = _interpreter(tflite_path)
    layers = []
    for op in interpreter._get_ops_details():
        if op["op_name"] == "FULLY_CONNECTED":
            # FULLY_CONNECTED inputs are [activation, weights, bias];
            # TFLite stores weights as (out, in), Keras as (in, out).
            _, w_index, b_index = op["inputs"]
            layers.append((interpreter.get_tensor(w_index).T, interpreter.get_tensor(b_index)))

    model = build_keras_model(model_cfg["input_size"], model_cfg["hidden_size"], model_cfg["bottleneck_size"])
    dense_layers = [layer for layer in model.layers if layer.weights]
    if len(dense_layers) != len(layers):
        raise ValueError(f"expected {len(dense_layers)} dense layers, found {len(layers)} in {tflite_path}")
    for layer, weights in zip(dense_layers, layers):
        layer.set_weights(list(weights))
    return model


def scores(interpreter, vectors: np.ndarray) -> np.ndarray:
    return np.array([compute_anomaly_score(v, reconstruct(interpreter, v)) for v in vectors])


def export_int8(keras_model, scaled_windows: np.ndarray, sigma: float, per_tensor: bool = True) -> dict:
    """Convert to int8 and calibrate its threshold on the same normal windows as the float model.

    WHY a separate threshold: quantization adds a small rounding error to
    every reconstruction, so "normal" errors are slightly larger. Reusing the
    float threshold would raise false alarms.
    """
    convert_to_tflite_int8(keras_model, INT8_MODEL, scaled_windows, per_tensor=per_tensor)
    mse = scores(_interpreter(INT8_MODEL), scaled_windows)
    threshold = compute_threshold(mse, sigma)
    with open(INT8_THRESHOLD, "w", encoding="utf-8") as f:
        json.dump(threshold, f, indent=2)
    return threshold


def describe_quantization(path: str) -> None:
    interpreter = _interpreter(path)
    for name, detail in (("input", interpreter.get_input_details()[0]), ("output", interpreter.get_output_details()[0])):
        scale, zero_point = detail["quantization"]
        lo, hi = (-128 - zero_point) * scale, (127 - zero_point) * scale
        print(f"  {name:6s} {detail['dtype'].__name__}  scale={scale:.6f}  zero_point={zero_point}  "
              f"covers [{lo:.3f}, {hi:.3f}]")
    for op in interpreter._get_ops_details():
        if op["op_name"] == "FULLY_CONNECTED":
            w = interpreter.get_tensor_details()[op["inputs"][1]]
            n_scales = len(w["quantization_parameters"]["scales"])
            kind = "per-tensor" if n_scales == 1 else f"per-channel ({n_scales} scales)"
            print(f"  weights {str(tuple(w['shape'])):10s} {w['dtype'].__name__}  {kind}")


def compare(name: str, vectors: np.ndarray, labels, float_model, float_thr: float, int8_model, int8_thr: float):
    f_scores, q_scores = scores(float_model, vectors), scores(int8_model, vectors)
    f_flags, q_flags = f_scores > float_thr, q_scores > int8_thr
    print(f"\n{name}: {len(vectors)} vectors")
    print(f"  score correlation float vs int8 : {np.corrcoef(f_scores, q_scores)[0, 1]:.5f}")
    print(f"  median |score difference|       : {np.median(np.abs(f_scores - q_scores)):.6f}")
    print(f"  decisions that differ          : {int(np.sum(f_flags != q_flags))}")
    if labels is not None:
        for label, flags in (("float32", f_flags), ("int8", q_flags)):
            tp = int(np.sum(flags & (labels == 1)))
            fp = int(np.sum(flags & (labels == 0)))
            print(f"  {label:7s} caught {tp}/{int(np.sum(labels == 1))} anomalies, "
                  f"{fp}/{int(np.sum(labels == 0))} false alarms")
    return int(np.sum(f_flags != q_flags))


def test_windows(test: np.ndarray, window_size: int, seed: int = 7):
    """Labelled windows shaped like production input.

    Normal: the mean of window_size random normal test rows.
    Fault:  a normal window in which the tags corrupted in one anomalous row
            keep that row's value for the whole window (a sustained fault),
            so their mean equals the faulty value.
    WHY not single rows: the scaler was fitted on window means, and a raw row
    is about sqrt(window_size) times noisier than a mean, so even normal rows
    would look anomalous.
    """
    from generate_synthetic import CENTRES, NOISES

    rng = np.random.default_rng(seed)
    values, labels = test[:, :-1], test[:, -1].astype(int)
    normal_rows, fault_rows = values[labels == 0], values[labels == 1]

    def normal_window():
        return normal_rows[rng.choice(len(normal_rows), size=window_size, replace=False)].mean(axis=0)

    normals = [normal_window() for _ in range(len(fault_rows) * 2)]
    faults = []
    for row in fault_rows:
        window = normal_window()
        # Normal values stay within +/- noise of the centre; injected ones are 3-5 noise away.
        corrupted = np.abs(row - CENTRES) > NOISES
        window[corrupted] = row[corrupted]
        faults.append(window)
    windows = np.array(normals + faults, dtype=np.float32)
    return windows, np.array([0] * len(normals) + [1] * len(faults))


def main():
    per_tensor = "--per-channel" not in sys.argv[1:]
    with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_cfg, prep_cfg = config["model"], config["preprocessing"]
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "threshold.json"), "r", encoding="utf-8") as f:
        float_threshold = json.load(f)

    train_windows = simulate_windowing(load_csv(os.path.join(BASE_DIR, "data", "normal_train.csv")),
                                       prep_cfg["window_size"])
    scaled_train = scaler.transform(train_windows).astype(np.float32)

    float_path = os.path.join(MODELS_DIR, "autoencoder.tflite")
    keras_model = keras_from_tflite(float_path, model_cfg)
    float_model = _interpreter(float_path)
    rebuild_error = np.max(np.abs(keras_model.predict(scaled_train, verbose=0)
                                  - np.array([reconstruct(float_model, v) for v in scaled_train])))
    print(f"Rebuilt Keras model vs float .tflite, max output difference: {rebuild_error:.2e}")
    if rebuild_error > 1e-4:
        raise SystemExit("Rebuild does not match the float model; refusing to quantize.")

    int8_threshold = export_int8(keras_model, scaled_train, float_threshold["sigma"], per_tensor=per_tensor)
    int8_model = _interpreter(INT8_MODEL)

    print(f"\nSaved {INT8_MODEL} ({os.path.getsize(INT8_MODEL)} bytes, "
          f"float model {os.path.getsize(float_path)} bytes)")
    print(f"Thresholds: float32 {float_threshold['threshold']:.6f}   int8 {int8_threshold['threshold']:.6f}")
    print("Quantization of the int8 model:")
    describe_quantization(INT8_MODEL)

    test = load_csv(os.path.join(BASE_DIR, "data", "test_mixed.csv"))
    windows, window_labels = test_windows(test, prep_cfg["window_size"])
    args = (float_model, float_threshold["threshold"], int8_model, int8_threshold["threshold"])
    compare("Normal training windows", scaled_train, None, *args)
    flips = compare("Labelled test windows (normal + sustained faults)",
                    scaler.transform(windows).astype(np.float32), window_labels, *args)

    print("\nGATE:", "PASS, int8 makes the same decisions" if flips == 0
          else f"REVIEW, {flips} test decisions differ between float32 and int8")


if __name__ == "__main__":
    main()
