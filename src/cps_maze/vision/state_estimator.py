from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallState:
    position_mm: np.ndarray
    velocity_mm_s: np.ndarray
    timestamp_s: float


class LowPassVelocityEstimator:
    def __init__(self, alpha: float = 0.35):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.previous_position: np.ndarray | None = None
        self.previous_timestamp_s: float | None = None
        self.velocity_mm_s = np.zeros(2, dtype=float)

    def reset(self) -> None:
        self.previous_position = None
        self.previous_timestamp_s = None
        self.velocity_mm_s = np.zeros(2, dtype=float)

    def update(self, position_mm: np.ndarray, timestamp_s: float) -> BallState:
        if self.previous_position is not None and self.previous_timestamp_s is not None:
            dt = max(timestamp_s - self.previous_timestamp_s, 1e-3)
            measured_velocity = (position_mm - self.previous_position) / dt
            self.velocity_mm_s = (
                self.alpha * measured_velocity + (1.0 - self.alpha) * self.velocity_mm_s
            )

        self.previous_position = position_mm
        self.previous_timestamp_s = timestamp_s
        return BallState(
            position_mm=position_mm,
            velocity_mm_s=self.velocity_mm_s.copy(),
            timestamp_s=timestamp_s,
        )
