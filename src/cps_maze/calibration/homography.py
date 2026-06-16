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

