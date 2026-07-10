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


def test_windowed_progress_stays_in_own_corridor():
    # Two corridors 12mm apart (a wall's width in a snaking maze); global
    # nearest-segment search would snap to the far corridor once the ball
    # drifts closer to the wall than to its own centerline.
    path = WaypointPath(points_mm=np.array(
        [[20.0, 50.0], [200.0, 50.0], [200.0, 62.0], [20.0, 62.0]]))
    ball = np.array([100.0, 57.0])  # 7mm from own line, 5mm from the far one

    global_progress = path.nearest_progress_mm(ball)
    windowed_progress, _ = path.nearest_progress_and_distance_mm(
        ball, near_progress_mm=70.0
    )

    assert global_progress > 190  # snaps to the far corridor: the bug
    assert 60 < windowed_progress < 120  # stays in the near corridor: fixed


def test_heading_change_accumulates_through_a_chicane():
    # A chicane's opposite turns cancel at the endpoints; comparing only the
    # tangent 30mm ahead would read this as nearly straight and let the
    # controller barrel through at full speed.
    chicane = WaypointPath(points_mm=np.array(
        [[0.0, 20.0], [40.0, 20.0], [52.0, 32.0], [64.0, 20.0], [110.0, 20.0]]))

    turn = chicane.heading_change_deg(35.0)

    t0 = chicane.tangent_at_progress_mm(35.0)
    t1 = chicane.tangent_at_progress_mm(65.0)
    endpoint_only = float(np.degrees(np.arccos(np.clip(np.dot(t0, t1), -1.0, 1.0))))

    assert turn > 60.0
    assert turn > 2 * endpoint_only


def test_heading_change_ignores_annotation_zigzag_noise():
    # An auto-traced straight has small zigzags; accumulating their absolute
    # angles made straights read as phantom corners (random slowdowns).
    rng = np.random.default_rng(7)
    xs = np.arange(0.0, 120.0, 4.0)
    noisy_straight = np.column_stack([xs, 50.0 + rng.uniform(-0.4, 0.4, len(xs))])
    path = WaypointPath(points_mm=noisy_straight)

    turn = path.heading_change_deg(10.0, noise_deg=12.0)

    assert turn < 15.0, f"noisy straight must not read as a corner ({turn:.0f} deg)"


def test_heading_change_still_registers_real_corner_with_deadband():
    corner = WaypointPath(points_mm=np.array(
        [[0.0, 0.0], [40.0, 0.0], [40.0, 40.0]]))

    turn = corner.heading_change_deg(20.0, noise_deg=12.0)

    assert turn > 60.0
