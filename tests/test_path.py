import numpy as np

from cps_maze.planning.path import WaypointPath


def test_target_ahead_returns_future_point():
    path = WaypointPath(points_mm=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]))

    target = path.target_ahead(np.array([0.0, 0.0]), lookahead_mm=15.0)

    assert np.allclose(target, [15.0, 0.0])


def test_nearest_progress_projects_onto_segment():
    path = WaypointPath(points_mm=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]))

    progress = path.nearest_progress_mm(np.array([4.0, 3.0]))

    assert progress == 4.0
