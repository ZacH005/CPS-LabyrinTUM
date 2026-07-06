from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_MM = 12.0
CHARUCO_MARKER_LENGTH_MM = 9.0

# Maze-plane position of the printed pattern's outer top-left corner, in the
# maze frame (origin = play-area top-left corner, x right, y down - same
# convention as the corner-click calibration). The pattern must lie FLAT on
# the play surface, axis-aligned with the board edges. Re-measure whenever it
# moves, or override per run with calibrate_charuco_homography.py
# --pattern-x-mm / --pattern-y-mm. The (0, 0) default anchors the frame at the
# pattern itself, which is still a valid metric frame for all tools.
CHARUCO_BOARD_TOP_LEFT_MM = np.array([0.0, 0.0], dtype=float)


@dataclass(frozen=True)
class CharucoBoardGeometry:
    squares_x: int = CHARUCO_SQUARES_X
    squares_y: int = CHARUCO_SQUARES_Y
    square_length_mm: float = CHARUCO_SQUARE_LENGTH_MM
    marker_length_mm: float = CHARUCO_MARKER_LENGTH_MM

    @property
    def inner_corners_x(self) -> int:
        return self.squares_x - 1

    @property
    def inner_corners_y(self) -> int:
        return self.squares_y - 1

    @property
    def board_size_mm(self) -> float:
        return self.squares_x * self.square_length_mm


def board_charuco_corner_points_mm(geometry: CharucoBoardGeometry | None = None) -> np.ndarray:
    geometry = geometry or CharucoBoardGeometry()
    x_coords = np.arange(1, geometry.squares_x) * geometry.square_length_mm
    y_coords = np.arange(1, geometry.squares_y) * geometry.square_length_mm
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)


def board_charuco_corner_points_to_maze_mm(
    board_points_mm: np.ndarray,
    board_top_left_mm: np.ndarray | None = None,
) -> np.ndarray:
    board_top_left_mm = CHARUCO_BOARD_TOP_LEFT_MM if board_top_left_mm is None else board_top_left_mm
    # y grows DOWN, matching the corner-click calibration frame so both
    # calibrations produce interchangeable coordinates.
    maze_points = np.empty_like(board_points_mm, dtype=np.float32)
    maze_points[:, 0] = board_top_left_mm[0] + board_points_mm[:, 0]
    maze_points[:, 1] = board_top_left_mm[1] + board_points_mm[:, 1]
    return maze_points


def charuco_ids_to_maze_points_mm(
    charuco_ids: np.ndarray,
    geometry: CharucoBoardGeometry | None = None,
    board_top_left_mm: np.ndarray | None = None,
) -> np.ndarray:
    geometry = geometry or CharucoBoardGeometry()
    board_points = board_charuco_corner_points_mm(geometry)
    maze_points = board_charuco_corner_points_to_maze_mm(board_points, board_top_left_mm)

    ids = np.asarray(charuco_ids, dtype=int).reshape(-1)
    if ids.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if ids.min(initial=0) < 0 or ids.max(initial=-1) >= len(maze_points):
        raise ValueError("charuco_ids contain points outside the configured board geometry")
    return maze_points[ids]