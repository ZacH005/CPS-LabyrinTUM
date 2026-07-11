import numpy as np

from cps_maze.planning.hazards import HoleMap
from cps_maze.planning.path import WaypointPath
from cps_maze.planning.speed_profile import build_speed_profile


def _straight_path(length_mm: float = 300.0) -> WaypointPath:
    return WaypointPath(points_mm=np.array([[0.0, 0.0], [length_mm, 0.0]]))


def test_profile_cruises_on_clear_straight():
    profile = build_speed_profile(
        _straight_path(), hole_map=None, wall_map=None,
        v_max_mm_s=25.0, accel_mm_s2=150.0)

    assert np.isclose(profile.speed_at(150.0), 25.0)


def test_profile_commits_to_pass_speed_through_hole_zone():
    # hole capture zone overlapping the route: the plan asks for the
    # committed PASS speed there - not a crawl at the stall threshold
    holes = HoleMap(np.array([[150.0, 0.0, 8.0]]),
                    ball_radius_mm=6.0, margin_mm=4.0)
    profile = build_speed_profile(
        _straight_path(), holes, None,
        v_max_mm_s=25.0, hole_pass_mm_s=16.0, accel_mm_s2=150.0)

    assert np.isclose(profile.speed_at(150.0), 16.0, atol=0.6)
    # and the whole plan never asks for anything near the old 8 mm/s crawl
    assert profile.min_speed() >= 9.5  # only the goal-end dips below floor


def test_profile_brakes_before_the_pass_not_at_it():
    holes = HoleMap(np.array([[150.0, 0.0, 8.0]]),
                    ball_radius_mm=6.0, margin_mm=4.0)
    profile = build_speed_profile(
        _straight_path(), holes, None,
        v_max_mm_s=45.0, hole_pass_mm_s=16.0, hole_slow_band_mm=20.0,
        accel_mm_s2=150.0)

    # deceleration must begin upstream: at the physics-required braking
    # distance (45^2-16^2)/(2*150) = 5.9mm before the slow band starts
    v_early = profile.speed_at(60.0)
    v_approach = profile.speed_at(125.0)
    v_pass = profile.speed_at(150.0)
    assert v_early > 40.0
    assert v_pass < v_approach < v_early


def test_profile_is_braking_and_acceleration_feasible_everywhere():
    holes = HoleMap(np.array([[100.0, 0.0, 8.0], [180.0, 4.0, 8.0]]),
                    ball_radius_mm=6.0, margin_mm=4.0)
    a = 150.0
    profile = build_speed_profile(
        _straight_path(), holes, None,
        v_max_mm_s=45.0, accel_mm_s2=a, step_mm=2.0)

    v = profile.speeds_mm_s
    dv2 = np.diff(v ** 2)
    limit = 2.0 * a * 2.0 + 1e-6
    assert np.all(dv2 <= limit), "acceleration exceeds the achievable limit"
    assert np.all(-dv2 <= limit), "braking exceeds the achievable limit"


def test_profile_on_real_maze_has_no_stall_trap():
    """Regression for the observed 'spazzing': on the real route + holes the
    plan must never ask for a speed at/below the stall-detection region, and
    must stay smooth through the overlapping capture zones."""
    path = WaypointPath.from_csv("configs/maze_path_auto.csv")
    holes_arr = np.genfromtxt("configs/maze_holes.csv", delimiter=",", names=True)
    holes = HoleMap(np.column_stack([holes_arr["x_mm"], holes_arr["y_mm"],
                                     holes_arr["radius_mm"]]),
                    ball_radius_mm=6.0, margin_mm=4.0)
    profile = build_speed_profile(
        path, holes, None,
        v_max_mm_s=25.0, hole_pass_mm_s=16.0, floor_mm_s=12.0,
        accel_mm_s2=118.0, end_speed_mm_s=10.0)

    v = profile.speeds_mm_s
    # everywhere except the goal approach: at least the floor
    assert float(np.min(v[:-10])) >= 11.5
    # a sane stall threshold fits below the plan minimum with margin
    assert 0.5 * profile.min_speed() >= 4.5
    # feasibility on the real route too
    dv2 = np.diff(v ** 2)
    limit = 2.0 * 118.0 * profile.step_mm + 1e-6
    assert np.all(np.abs(dv2) <= limit)
