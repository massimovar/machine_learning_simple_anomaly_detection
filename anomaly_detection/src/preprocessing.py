import pickle
from typing import List, Optional

import numpy as np


class PreprocessingError(ValueError):
    pass


class SampleBuffer:
    def __init__(self, window_size: int, tag_names: List[str]):
        if window_size <= 0:
            raise PreprocessingError("window_size must be greater than zero")
        if not tag_names:
            raise PreprocessingError("tag_names cannot be empty")

        self.window_size = window_size
        self.tag_names = list(tag_names)
        self.samples: List[List[float]] = []

    def add(self, sensor_dict: dict) -> Optional[np.ndarray]:
        try:
            row = [float(sensor_dict[tag]) for tag in self.tag_names]
        except KeyError as exc:
            raise PreprocessingError(f"missing sensor tag: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise PreprocessingError("sensor values must be numeric") from exc

        self.samples.append(row)

        if len(self.samples) == self.window_size:
            window = np.asarray(self.samples, dtype=np.float32)
            self.samples.clear()
            return window

        return None


def aggregate_window(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    if window.ndim != 2 or window.shape[0] == 0:
        raise PreprocessingError("window must be a non-empty 2-D array")
    return window.mean(axis=0)


def load_scaler(scaler_path: str):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def normalise(feature_vector: np.ndarray, scaler) -> np.ndarray:
    feature_vector = np.asarray(feature_vector, dtype=np.float32)
    if feature_vector.ndim != 1:
        raise PreprocessingError("feature_vector must be a 1-D array")
    return scaler.transform(feature_vector.reshape(1, -1)).flatten()
