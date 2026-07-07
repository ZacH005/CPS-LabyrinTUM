from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AxisMap:
    """Maps a board-frame command (cx, cy) to a servo command (yaw, pitch).

    The controller thinks in board coordinates: "push the ball toward +x".
    Which servo channel tilts which board axis, and with which sign, is a
    property of the physical build (mounting orientation, linkage direction,
    connector channel order). That mapping is measured once by
    scripts/axis_check.py and stored here, so the control code never needs
    hand-flipped signs.
    """

    matrix: np.ndarray  # shape (2, 2): (yaw, pitch) = matrix @ (cx, cy)

    def __post_init__(self) -> None:
        if self.matrix.shape != (2, 2):
            raise ValueError("axis map matrix must have shape (2, 2)")
        if abs(float(np.linalg.det(self.matrix))) < 1e-9:
            raise ValueError("axis map matrix is singular")

    @classmethod
    def identity(cls) -> "AxisMap":
        return cls(matrix=np.eye(2))

    def apply(self, board_command: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(board_command, dtype=float)

    def save(self, path: str | Path) -> None:
        np.savez(Path(path), matrix=self.matrix)

    @classmethod
    def load(cls, path: str | Path) -> "AxisMap":
        data = np.load(Path(path))
        return cls(matrix=data["matrix"].astype(float))


def snap_response_to_axis_map(response: np.ndarray) -> AxisMap:
    """Derive a clean axis map from a measured response matrix.

    ``response`` columns are the ball displacement (dx, dy) in board mm caused
    by a unit +yaw and a unit +pitch command respectively. The map is snapped
    to a permutation/sign matrix (each servo drives exactly one board axis),
    which is the physically expected structure and is robust to measurement
    noise in the off-axis terms.
    """
    if response.shape != (2, 2):
        raise ValueError("response must have shape (2, 2)")

    snapped = np.zeros((2, 2))
    for j in range(2):  # j: command axis (0=yaw, 1=pitch)
        i = int(np.argmax(np.abs(response[:, j])))  # dominant board axis
        snapped[i, j] = np.sign(response[i, j])

    if abs(float(np.linalg.det(snapped))) < 1e-9:
        raise ValueError(
            "Both servo commands moved the same board axis - check wiring/"
            "linkage, or rerun with a larger pulse. Measured response:\n"
            f"{response}"
        )

    # snapped maps servo->board; the controller needs board->servo.
    return AxisMap(matrix=np.linalg.inv(snapped))
