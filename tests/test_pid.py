import numpy as np

from cps_maze.control.pid import (
    PathFollower,
    PathFollowerConfig,
    VelocityFollowerConfig,
    VelocityPathFollower,
)


def test_path_follower_clamps_command():
    follower = PathFollower(PathFollowerConfig(kp=10.0, kd=0.0, max_command=1.0))

    command = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([0.0, 0.0]),
        target_mm=np.array([10.0, -10.0]),
    )

    assert np.allclose(command, [1.0, -1.0])


def test_path_follower_stall_kick_ignores_single_slow_frame():
    # A ball easing into a target reads "slow" for a moment; that alone must
    # not trigger the full stiction kick, or ordinary braking gets punched.
    follower = PathFollower(PathFollowerConfig(
        kp=0.01, kd=0.0, max_command=1.0,
        stall_kick=0.5, stall_speed_mm_s=8.0, stall_dist_mm=5.0,
        stall_min_duration_s=0.3,
    ))

    command = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([1.0, 0.0]),  # below stall_speed_mm_s
        target_mm=np.array([50.0, 0.0]),
        dt_s=0.02,  # one frame: far short of stall_min_duration_s
    )

    unkicked = 0.01 * 50.0
    assert np.isclose(command[0], unkicked)


def test_path_follower_stall_kick_fires_after_sustained_stop():
    follower = PathFollower(PathFollowerConfig(
        kp=0.01, kd=0.0, max_command=1.0,
        stall_kick=0.5, stall_speed_mm_s=8.0, stall_dist_mm=5.0,
        stall_min_duration_s=0.3,
    ))

    command = None
    for _ in range(20):  # 20 * 0.02s = 0.4s of continuous low speed
        command = follower.command(
            position_mm=np.array([0.0, 0.0]),
            velocity_mm_s=np.array([1.0, 0.0]),
            target_mm=np.array([50.0, 0.0]),
            dt_s=0.02,
        )

    assert np.isclose(command[0], 0.5)


def test_velocity_follower_stall_kick_requires_persistence():
    cfg = VelocityFollowerConfig(
        v_max_mm_s=30.0, k_lat=0.0, k_vel=0.001,
        max_command=1.0, stall_kick=0.5, stall_speed_mm_s=8.0,
        stall_min_duration_s=0.3,
    )
    follower = VelocityPathFollower(cfg)
    path_point = np.array([0.0, 0.0])
    tangent = np.array([1.0, 0.0])

    # single slow frame: no kick, tiny k_vel keeps the raw command small
    command, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([1.0, 0.0]),
        path_point_mm=path_point, tangent=tangent,
        heading_change_deg=0.0, dt_s=0.02,
    )
    assert abs(command[0]) < 0.05

    # sustained low speed: the kick engages
    for _ in range(20):
        command, _ = follower.command(
            position_mm=np.array([0.0, 0.0]),
            velocity_mm_s=np.array([1.0, 0.0]),
            path_point_mm=path_point, tangent=tangent,
            heading_change_deg=0.0, dt_s=0.02,
        )
    assert np.isclose(command[0], 0.5)


def test_velocity_follower_reset_clears_persistence_timer():
    cfg = VelocityFollowerConfig(
        v_max_mm_s=30.0, k_lat=0.0, k_vel=0.001,
        max_command=1.0, stall_kick=0.5, stall_speed_mm_s=8.0,
        stall_min_duration_s=0.3,
    )
    follower = VelocityPathFollower(cfg)
    path_point = np.array([0.0, 0.0])
    tangent = np.array([1.0, 0.0])

    for _ in range(20):
        follower.command(
            position_mm=np.array([0.0, 0.0]),
            velocity_mm_s=np.array([1.0, 0.0]),
            path_point_mm=path_point, tangent=tangent,
            heading_change_deg=0.0, dt_s=0.02,
        )
    follower.reset()

    command, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([1.0, 0.0]),
        path_point_mm=path_point, tangent=tangent,
        heading_change_deg=0.0, dt_s=0.02,
    )
    assert abs(command[0]) < 0.05

