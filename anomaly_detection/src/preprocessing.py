"""
preprocessing.py — Windowing and normalisation
===============================================

WHY THIS FILE EXISTS:
    Raw sensor readings arrive one at a time (1 message per second).
    Before we can feed them to the autoencoder, we need to:

    1. **Buffer** them into windows (groups of N samples).
    2. **Aggregate** each window into a single feature vector (we take the mean
       of each tag over the window).
    3. **Normalise** the feature vector to [0, 1] using a pre-fitted scaler.

    This keeps the ML model simple: it always sees one fixed-size vector of 10
    numbers, each between 0 and 1.

KEY CONCEPTS:
    - Windowing:  Collecting N consecutive samples and treating them as one unit.
                  We use non-overlapping windows (no overlap) for simplicity.
    - MinMaxScaler:  Rescales each feature so that its minimum (from training)
                  maps to 0 and its maximum maps to 1.  This prevents tags with
                  large values (e.g., cycle_time ~200 ms) from dominating tags
                  with small values (e.g., vibration_rms ~2 mm/s).
"""

import pickle
from typing import List, Optional

import numpy as np


class SampleBuffer:
    """
    Collects individual sensor samples and yields a window when full.

    Usage:
        buffer = SampleBuffer(window_size=30, tag_names=[...])
        for each incoming MQTT message:
            window = buffer.add(parsed_dict)
            if window is not None:
                # window is a numpy array of shape (30, 10)
                ...
    """

    def __init__(self, window_size: int, tag_names: List[str]):
        self.window_size = window_size
        self.tag_names = tag_names
        self.samples: List[List[float]] = []  # accumulates rows of sensor values

    def add(self, sensor_dict: dict) -> Optional[np.ndarray]:
        """
        Add one sensor reading (a dict like {"wire_feed_speed": 10.2, ...}).

        Returns
        -------
        np.ndarray or None
            If the buffer is now full, returns the window as a (window_size, n_tags)
            array and resets the buffer.  Otherwise returns None.
        """
        # Extract the tag values in the correct order
        row = [sensor_dict[tag] for tag in self.tag_names]
        self.samples.append(row)

        if len(self.samples) >= self.window_size:
            # Convert list-of-lists to a 2-D numpy array: shape (window_size, n_tags)
            window = np.array(self.samples, dtype=np.float32)
            self.samples = []  # reset for the next window
            return window

        return None


def aggregate_window(window: np.ndarray) -> np.ndarray:
    """
    Collapse a window of shape (window_size, n_tags) into a single vector of
    shape (n_tags,) by taking the **mean** of each tag across the window.

    WHY averaging:
        A 30-second average smooths out sensor noise while still capturing
        sustained anomalies (e.g., motor current drifting upward).  It also
        means our autoencoder only needs 10 inputs, not 300 (30×10), keeping
        the model tiny and fast.
    """
    # axis=0 means "average down the rows", leaving one value per column (tag)
    return window.mean(axis=0)


def load_scaler(scaler_path: str):
    """
    Load a pre-fitted MinMaxScaler from a pickle file.

    The scaler was fitted on normal training data (in train.py) and saved to
    disk so that at inference time we apply the exact same transformation.

    Returns
    -------
    sklearn.preprocessing.MinMaxScaler
    """
    with open(scaler_path, "rb") as f:
        # pickle.load deserialises the Python object that was saved during training.
        return pickle.load(f)


def normalise(feature_vector: np.ndarray, scaler) -> np.ndarray:
    """
    Apply the fitted MinMaxScaler to a single feature vector.

    Parameters
    ----------
    feature_vector : np.ndarray
        Shape (n_tags,) — e.g., [10.2, 22.5, 2.1, ...].
    scaler : MinMaxScaler
        Pre-fitted scaler loaded from disk.

    Returns
    -------
    np.ndarray
        Shape (n_tags,), values in [0, 1].
    """
    # scaler.transform expects a 2-D array: (n_samples, n_features).
    # We reshape our 1-D vector to (1, n_tags), transform, then flatten back.
    scaled = scaler.transform(feature_vector.reshape(1, -1))
    return scaled.flatten()
