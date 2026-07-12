"""Neutral trim: the command offset at which the board is actually level.

The servo neutral (command 0,0) only levels the BOARD if the table, frame,
and linkage are perfectly square - they are not, so at "neutral" the ball
drifts and the controller wastes integral headroom fighting a constant bias.

The trim is measured once with the ball itself as the level sensor
(scripts/calibrate_neutral_trim.py) and stored in a small JSON file. The
serial link adds it to every outgoing command, so zero means "level" for
every tool without any of them knowing about the slant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_TRIM_FILE = "calibration/neutral_trim.json"


@dataclass(frozen=True)
class NeutralTrim:
    yaw: float = 0.0
    pitch: float = 0.0

    def save(self, path: str | Path, residual_drift_mm_s: float | None = None) -> None:
        payload = {"yaw": self.yaw, "pitch": self.pitch}
        if residual_drift_mm_s is not None:
            payload["residual_drift_mm_s"] = residual_drift_mm_s
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "NeutralTrim":
        data = json.loads(Path(path).read_text())
        return cls(yaw=float(data["yaw"]), pitch=float(data["pitch"]))

    @classmethod
    def load_if_exists(cls, path: str | Path = DEFAULT_TRIM_FILE) -> "NeutralTrim":
        p = Path(path)
        return cls.load(p) if p.exists() else cls()


def trim_step(
    drift_mm_s: np.ndarray,
    axis_map,
    gain_cmd_per_mm_s: float = 0.004,
    max_step: float = 0.05,
) -> np.ndarray:
    """Servo-frame trim correction that opposes an observed ball drift.

    The ball drifting toward +x means the board is tilted toward +x, so the
    correction is a board-frame command against the drift, converted to the
    servo frame through the measured axis map, and step-limited so a noisy
    measurement cannot slingshot the search.
    """
    board_correction = -gain_cmd_per_mm_s * np.asarray(drift_mm_s, dtype=float)
    servo_delta = axis_map.apply(board_correction)
    magnitude = float(np.linalg.norm(servo_delta))
    if magnitude > max_step:
        servo_delta = servo_delta * (max_step / magnitude)
    return servo_delta
