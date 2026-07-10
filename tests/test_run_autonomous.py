import numpy as np

from cps_maze.planning.path import WaypointPath
from scripts.run_autonomous import choose_carrot_point


class XLimitWallMap:
    def __init__(self, max_clear_x: float):
        self.max_clear_x = max_clear_x

    def line_blocked(self, _a_mm: np.ndarray, b_mm: np.ndarray) -> bool:
        return bool(b_mm[0] > self.max_clear_x)


def test_choose_carrot_point_without_wall_map_uses_full_lookahead():
    path = WaypointPath(np.array([[0.0, 0.0], [50.0, 0.0]]))

    carrot, lookahead = choose_carrot_point(
        path=path,
        position_mm=np.array([0.0, 0.0]),
        progress_mm=0.0,
        lookahead_mm=30.0,
        min_lookahead_mm=10.0,
        wall_map=None,
    )

    assert np.allclose(carrot, [30.0, 0.0])
    assert np.isclose(lookahead, 30.0)


def test_choose_carrot_point_backs_down_to_clear_line_of_sight():
    path = WaypointPath(np.array([[0.0, 0.0], [50.0, 0.0]]))

    carrot, lookahead = choose_carrot_point(
        path=path,
        position_mm=np.array([0.0, 0.0]),
        progress_mm=0.0,
        lookahead_mm=30.0,
        min_lookahead_mm=10.0,
        wall_map=XLimitWallMap(max_clear_x=12.0),
        step_mm=5.0,
    )

    assert np.allclose(carrot, [10.0, 0.0])
    assert np.isclose(lookahead, 10.0)
