from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Homography:
    image_to_board: np.ndarray

    def image_point_to_board_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        point = np.array([[[x_px, y_px]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.image_to_board)
        x_mm, y_mm = transformed[0, 0]
        return float(x_mm), float(y_mm)

    def board_point_to_image_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        inverse = np.linalg.inv(self.image_to_board)
        point = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, inverse)
        x_px, y_px = transformed[0, 0]
        return float(x_px), float(y_px)

    def board_points_to_image_px(self, points_mm: np.ndarray) -> np.ndarray:
        """Vectorized board mm -> image px for overlay drawing. points_mm shape (N, 2)."""
        inverse = np.linalg.inv(self.image_to_board)
        pts = points_mm.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, inverse).reshape(-1, 2)

    def save(self, path: str | Path) -> None:
        np.savez(Path(path), image_to_board=self.image_to_board)

    @classmethod
    def load(cls, path: str | Path) -> "Homography":
        data = np.load(Path(path))
        return cls(image_to_board=data["image_to_board"])


def estimate_homography(
    image_points_px: np.ndarray,
    board_points_mm: np.ndarray,
) -> Homography:
    matrix, status = cv2.findHomography(
        image_points_px.astype(np.float32),
        board_points_mm.astype(np.float32),
    )
    if matrix is None or status is None:
        raise RuntimeError("Could not estimate homography")
    return Homography(image_to_board=matrix)

