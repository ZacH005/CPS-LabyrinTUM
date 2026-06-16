from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WaypointPath:
    points_mm: np.ndarray

    def __post_init__(self) -> None:
        if self.points_mm.ndim != 2 or self.points_mm.shape[1] != 2:
            raise ValueError("points_mm must have shape (N, 2)")
        if self.points_mm.shape[0] < 2:
            raise ValueError("Path requires at least two waypoints")

    @classmethod
    def from_csv(cls, path: str | Path) -> "WaypointPath":
        rows = np.genfromtxt(Path(path), delimiter=",", names=True)
        points = np.column_stack([rows["x_mm"], rows["y_mm"]]).astype(float)
        return cls(points_mm=points)

    @property
    def segment_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.points_mm, axis=0), axis=1)

    @property
    def cumulative_lengths(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.segment_lengths)])

    def nearest_progress_mm(self, position_mm: np.ndarray) -> float:
        best_distance = float("inf")
        best_progress = 0.0
        cumulative = self.cumulative_lengths

        for index, (start, end) in enumerate(zip(self.points_mm[:-1], self.points_mm[1:])):
            segment = end - start
            length_sq = float(np.dot(segment, segment))
            if length_sq == 0.0:
                continue
            t = float(np.clip(np.dot(position_mm - start, segment) / length_sq, 0.0, 1.0))
            projection = start + t * segment
            distance = float(np.linalg.norm(position_mm - projection))
            if distance < best_distance:
                best_distance = distance
                best_progress = float(cumulative[index] + t * np.sqrt(length_sq))

        return best_progress

    def point_at_progress_mm(self, progress_mm: float) -> np.ndarray:
        cumulative = self.cumulative_lengths
        total = float(cumulative[-1])
        progress = float(np.clip(progress_mm, 0.0, total))

        segment_index = int(np.searchsorted(cumulative, progress, side="right") - 1)
        segment_index = min(segment_index, len(self.points_mm) - 2)
        start_progress = cumulative[segment_index]
        segment_length = max(self.segment_lengths[segment_index], 1e-9)
        t = (progress - start_progress) / segment_length
        return self.points_mm[segment_index] + t * (
            self.points_mm[segment_index + 1] - self.points_mm[segment_index]
        )

    def target_ahead(self, position_mm: np.ndarray, lookahead_mm: float) -> np.ndarray:
        progress = self.nearest_progress_mm(position_mm)
        return self.point_at_progress_mm(progress + lookahead_mm)
