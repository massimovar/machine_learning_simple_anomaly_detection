"""
train.py — train the autoencoder and export both model files

OUTPUT (in models/):
    autoencoder.tflite        float32 model (CPU)
    autoencoder_int8.tflite   full-int8 twin of the same weights (NPU)
    threshold.json            three-sigma threshold of the float32 model
    threshold_int8.json       the same, calibrated on the int8 model
    scaler.json               per-tag min/max of the training windows

RUN (from anomaly_detection/, needs full TensorFlow):
    python training/generate_synthetic.py
    python training/train.py
"""

import json
import os
import sys

import numpy as np
import yaml

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from model import InferenceEngine  # noqa: E402


def load_csv(filepath: str) -> np.ndarray:
    return np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)


def simulate_windowing(data: np.ndarray, window_size: int) -> np.ndarray:
    """The detector's input: the mean of every window_size consecutive rows."""
    n_windows = len(data) // window_size
    return data[: n_windows * window_size].reshape(n_windows, window_size, -1).mean(axis=1)


def compute_threshold(mse_per_sample: np.ndarray, sigma: float) -> dict:
    mse_mean = float(np.mean(mse_per_sample))
    mse_std = float(np.std(mse_per_sample))
    return {"threshold": mse_mean + sigma * mse_std, "mean": mse_mean, "std": mse_std, "sigma": sigma}


def save_json(name: str, data: dict) -> None:
    with open(os.path.join(MODELS_DIR, name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_keras_model(input_size: int, hidden_size: int, bottleneck_size: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_size,), name="sensor_input")
    encoded = tf.keras.layers.Dense(hidden_size, activation="relu")(inputs)
    encoded = tf.keras.layers.Dense(bottleneck_size, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(hidden_size, activation="relu")(encoded)
    outputs = tf.keras.layers.Dense(input_size, activation="sigmoid")(decoded)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")


def convert(keras_model, representative_data=None) -> bytes:
    """float32 .tflite, or full int8 when representative_data is given.

    WHY full int8 (weights, activations, input and output): the i.MX 8M Plus
    NPU is an integer engine; a graph that still has float tensors is left
    (partly) on the CPU by the VX delegate. The representative data must look
    like production input (scaled window means); the converter uses it to
    choose each tensor's scale and zero point.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if representative_data is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: ([row.reshape(1, -1)] for row in representative_data)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        # WHY per-tensor: TFLite gives Dense weights one scale per output neuron
        # (per-channel) by default; the VIP8000 NPU is documented to prefer one
        # scale per tensor. Private flag (no public equivalent yet): the check
        # in main() makes a silent change visible.
        converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    return converter.convert()


def test_windows(test: np.ndarray, window_size: int, seed: int = 7):
    """Labelled windows shaped like production input.

    Normal: the mean of window_size random normal test rows.
    Fault:  a normal window in which the tags corrupted in one anomalous row
            keep that row's value for the whole window (a sustained fault).
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
    return np.array(normals + faults, dtype=np.float32), np.array([0] * len(normals) + [1] * len(faults))


def main():
    with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_cfg, prep_cfg = config["training"], config["preprocessing"]
    window_size, sigma = prep_cfg["window_size"], train_cfg["threshold_sigma"]
    os.makedirs(MODELS_DIR, exist_ok=True)

    windows = simulate_windowing(load_csv(os.path.join(BASE_DIR, "data", "normal_train.csv")), window_size)
    data_min, data_max = windows.min(axis=0), windows.max(axis=0)

    def scale(x):
        return ((x - data_min) / (data_max - data_min)).astype(np.float32)

    scaled = scale(windows)

    import tensorflow as tf

    model = build_keras_model(len(prep_cfg["tag_names"]), train_cfg["hidden_size"], train_cfg["bottleneck_size"])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]), loss="mse")
    model.fit(scaled, scaled, epochs=train_cfg["epochs"], batch_size=train_cfg["batch_size"], shuffle=True, verbose=1)

    float_path = os.path.join(MODELS_DIR, "autoencoder.tflite")
    int8_path = os.path.join(MODELS_DIR, "autoencoder_int8.tflite")
    with open(float_path, "wb") as f:
        f.write(convert(model))
    with open(int8_path, "wb") as f:
        f.write(convert(model, scaled))
    save_json("scaler.json", {"min": data_min.tolist(), "max": data_max.tolist()})

    float_mse = np.mean((scaled - model.predict(scaled, verbose=0)) ** 2, axis=1)
    float_threshold = compute_threshold(float_mse, sigma)
    save_json("threshold.json", float_threshold)
    # WHY a separate int8 threshold: quantization adds a small rounding error to
    # every reconstruction, so "normal" errors are slightly larger. Reusing the
    # float threshold would raise false alarms.
    float_engine, int8_engine = InferenceEngine(float_path), InferenceEngine(int8_path)
    int8_threshold = compute_threshold(np.array([int8_engine.score(v) for v in scaled]), sigma)
    save_json("threshold_int8.json", int8_threshold)

    print(f"\nSaved to {MODELS_DIR}: autoencoder.tflite, autoencoder_int8.tflite, "
          "threshold.json, threshold_int8.json, scaler.json")
    print(f"Thresholds: float32 {float_threshold['threshold']:.6f}   int8 {int8_threshold['threshold']:.6f}")

    per_channel = [d["name"] for d in int8_engine.interpreter.get_tensor_details()
                   if len(d["quantization_parameters"]["scales"]) > 1]
    if per_channel:
        print(f"WARNING: per-channel quantized tensors (the NPU prefers per-tensor): {per_channel}")

    # Same decisions on labelled windows? This is the gate for shipping the int8 model.
    vectors, labels = test_windows(load_csv(os.path.join(BASE_DIR, "data", "test_mixed.csv")), window_size)
    vectors = scale(vectors)
    flags = {}
    for name, engine, thr in (("float32", float_engine, float_threshold), ("int8", int8_engine, int8_threshold)):
        flags[name] = np.array([engine.score(v) for v in vectors]) > thr["threshold"]
        print(f"  {name:7s} caught {int(np.sum(flags[name] & (labels == 1)))}/{int(np.sum(labels == 1))} anomalies, "
              f"{int(np.sum(flags[name] & (labels == 0)))}/{int(np.sum(labels == 0))} false alarms")
    flips = int(np.sum(flags["float32"] != flags["int8"]))
    print("GATE:", "PASS, int8 makes the same decisions" if flips == 0
          else f"REVIEW, {flips} test decisions differ between float32 and int8")


if __name__ == "__main__":
    main()
