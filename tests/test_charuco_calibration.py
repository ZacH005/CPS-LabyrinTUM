import numpy as np

from cps_maze.calibration.charuco import charuco_ids_to_maze_points_mm


def test_charuco_ids_map_into_maze_millimeters():
    # y grows down, matching the corner-click calibration frame.
    # Default anchor (0, 0): coordinates are relative to the pattern corner.
    ids = np.array([[0], [3], [12], [15]])

    maze_points = charuco_ids_to_maze_points_mm(ids)

    assert np.allclose(
        maze_points,
        np.array(
            [
                [12.0, 12.0],
                [48.0, 12.0],
                [12.0, 48.0],
                [48.0, 48.0],
            ]
        ),
    )


def test_charuco_pattern_placement_offset():
    ids = np.array([[0]])

    maze_points = charuco_ids_to_maze_points_mm(
        ids, board_top_left_mm=np.array([100.0, 50.0])
    )

    assert np.allclose(maze_points, np.array([[112.0, 62.0]]))
