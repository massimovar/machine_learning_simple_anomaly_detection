import json

import numpy as np


class SampleBuffer:
    """Collects one row per sensor message; every window_size rows returns their per-tag mean."""

    def __init__(self, window_size: int, tag_names: list):
        self.window_size = window_size
        self.tag_names = list(tag_names)
        self.rows = []

    def add(self, sensor: dict):
        # A missing tag raises KeyError, a non-numeric value ValueError: the
        # message is dropped before it can touch the window.
        self.rows.append([float(sensor[tag]) for tag in self.tag_names])
        if len(self.rows) < self.window_size:
            return None
        window = np.asarray(self.rows, dtype=np.float32)
        self.rows.clear()
        return window.mean(axis=0)


def load_scaler(path: str) -> dict:
    """The per-tag min and max of the training windows, saved by training/train.py."""
    with open(path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    return {"min": np.asarray(scaler["min"], dtype=np.float64), "max": np.asarray(scaler["max"], dtype=np.float64)}


def normalise(feature_vector: np.ndarray, scaler: dict) -> np.ndarray:
    """Min-max scaling: the training range of every tag maps to [0, 1]."""
    return ((np.asarray(feature_vector, dtype=np.float64) - scaler["min"])
            / (scaler["max"] - scaler["min"])).astype(np.float32)
