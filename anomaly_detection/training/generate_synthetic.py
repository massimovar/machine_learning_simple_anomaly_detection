import csv
import os

import numpy as np

TAG_SPECS = [
    ("wire_feed_speed", 10.0, 2.0),
    ("wire_tension", 22.5, 7.5),
    ("motor_current_feed", 2.25, 0.75),
    ("motor_current_bend", 3.25, 1.25),
    ("bend_angle", 180.0, 2.0),
    ("cycle_time", 200.0, 20.0),
    ("vibration_rms", 1.75, 1.25),
    ("cut_force", 55.0, 15.0),
    ("temperature_motor", 47.5, 17.5),
    ("reject_count_per_min", 1.0, 1.0),
]

TAG_NAMES = [name for name, _, _ in TAG_SPECS]
CENTRES = np.array([centre for _, centre, _ in TAG_SPECS])
NOISES = np.array([noise for _, _, noise in TAG_SPECS])


def generate_normal_samples(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return CENTRES + rng.uniform(-NOISES, NOISES, size=(n, len(TAG_SPECS)))


def inject_anomalies(
    samples: np.ndarray,
    fraction: float = 0.10,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    corrupted = samples.copy()
    labels = np.zeros(len(samples), dtype=int)
    anomaly_indices = rng.choice(len(samples), size=int(len(samples) * fraction), replace=False)

    for row_idx in anomaly_indices:
        labels[row_idx] = 1
        tag_indices = rng.choice(len(TAG_SPECS), size=rng.integers(1, 4), replace=False)
        for tag_idx in tag_indices:
            direction = rng.choice([-1, 1])
            corrupted[row_idx, tag_idx] = CENTRES[tag_idx] + direction * rng.uniform(3.0, 5.0) * NOISES[tag_idx]

    return corrupted, labels


def save_csv(filepath: str, data: np.ndarray, headers: list[str]) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows([round(float(value), 4) for value in row] for row in data)


def main():
    normal_train = generate_normal_samples(5000, seed=42)
    save_csv("data/normal_train.csv", normal_train, TAG_NAMES)

    normal_test = generate_normal_samples(1000, seed=123)
    test_mixed, labels = inject_anomalies(normal_test, fraction=0.10, seed=99)
    save_csv("data/test_mixed.csv", np.column_stack([test_mixed, labels]), TAG_NAMES + ["is_anomaly"])

    print("Synthetic data saved in data/")


if __name__ == "__main__":
    main()
