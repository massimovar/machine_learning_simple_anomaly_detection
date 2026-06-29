import json
import os
import pickle
import sys

import numpy as np
import yaml

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from model import build_keras_model, convert_to_tflite  # noqa: E402


def load_csv(filepath: str) -> np.ndarray:
    return np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)


def simulate_windowing(data: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")

    n_windows = len(data) // window_size
    if n_windows == 0:
        raise ValueError("not enough rows for one window")

    trimmed = data[: n_windows * window_size]
    return trimmed.reshape(n_windows, window_size, -1).mean(axis=1)


def main():
    config_path = os.path.join(BASE_DIR, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]
    prep_cfg = config["preprocessing"]

    print("=" * 60)
    print("Training autoencoder")
    print("=" * 60)

    raw_data = load_csv(os.path.join(BASE_DIR, "data", "normal_train.csv"))
    windowed = simulate_windowing(raw_data, prep_cfg["window_size"])

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(windowed).astype(np.float32)

    model = build_keras_model(
        input_size=model_cfg["input_size"],
        hidden_size=model_cfg["hidden_size"],
        bottleneck_size=model_cfg["bottleneck_size"],
    )

    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]),
        loss="mse",
    )
    model.fit(
        x=scaled_data,
        y=scaled_data,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        verbose=1,
    )

    all_reconstructed = model.predict(scaled_data, verbose=0)
    mse_per_sample = np.mean((scaled_data - all_reconstructed) ** 2, axis=1)
    mse_mean = float(mse_per_sample.mean())
    mse_std = float(mse_per_sample.std())
    sigma = train_cfg["threshold_sigma"]
    threshold = mse_mean + sigma * mse_std

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    tflite_path = os.path.join(models_dir, "autoencoder.tflite")
    convert_to_tflite(model, tflite_path)

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    threshold_path = os.path.join(models_dir, "threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "mean": mse_mean, "std": mse_std, "sigma": sigma}, f, indent=2)

    print(f"Saved model to {tflite_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved threshold to {threshold_path}")


if __name__ == "__main__":
    main()
