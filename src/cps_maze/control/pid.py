from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PathFollowerConfig:
    kp: float
    kd: float
    max_command: float


class PathFollower:
    def __init__(self, config: PathFollowerConfig):
        self.config = config

    def command(
        self,
        position_mm: np.ndarray,
        velocity_mm_s: np.ndarray,
        target_mm: np.ndarray,
    ) -> np.ndarray:
        error = target_mm - position_mm
        raw = self.config.kp * error - self.config.kd * velocity_mm_s
        return np.clip(raw, -self.config.max_command, self.config.max_command)

