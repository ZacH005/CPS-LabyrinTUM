from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PathFollowerConfig:
    kp: float
    kd: float
    max_command: float
    # Integral action: accumulates while error persists, which walks the
    # command up until the ball un-sticks; also absorbs a non-level neutral.
    ki: float = 0.0
    integral_limit: float = 0.25  # max |command| contribution of the I term
    # Stiction kick: below some tilt the ball simply does not move (static
    # friction). If the ball is stalled away from the target, scale the
    # command up to at least this magnitude so it always breaks free.
    # Set from axis_check observations (~the amplitude that reliably moved
    # the ball). 0 disables.
    stall_kick: float = 0.0
    stall_speed_mm_s: float = 8.0   # "stalled" when slower than this
    stall_dist_mm: float = 8.0      # ...and further than this from target


class PathFollower:
    def __init__(self, config: PathFollowerConfig):
        self.config = config
        self.integral = np.zeros(2)

    def reset(self) -> None:
        self.integral = np.zeros(2)

    def command(
        self,
        position_mm: np.ndarray,
        velocity_mm_s: np.ndarray,
        target_mm: np.ndarray,
        dt_s: float = 0.0,
    ) -> np.ndarray:
        cfg = self.config
        error = target_mm - position_mm
        err_dist = float(np.linalg.norm(error))
        speed = float(np.linalg.norm(velocity_mm_s))

        if cfg.ki > 0.0 and dt_s > 0.0:
            self.integral += error * dt_s
            # anti-windup: clamp the I contribution, and bleed it off once
            # the ball is at the target so it cannot cause overshoot later
            limit = cfg.integral_limit / cfg.ki
            self.integral = np.clip(self.integral, -limit, limit)
            if err_dist < cfg.stall_dist_mm:
                self.integral *= 0.90

        raw = cfg.kp * error + cfg.ki * self.integral - cfg.kd * velocity_mm_s

        if (cfg.stall_kick > 0.0 and speed < cfg.stall_speed_mm_s
                and err_dist > cfg.stall_dist_mm):
            magnitude = float(np.linalg.norm(raw))
            if 1e-9 < magnitude < cfg.stall_kick:
                raw = raw * (cfg.stall_kick / magnitude)

        return np.clip(raw, -cfg.max_command, cfg.max_command)
