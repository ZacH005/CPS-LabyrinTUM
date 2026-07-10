import numpy as np

from cps_maze.control.pid import (
    CarrotVelocityFollowerConfig,
    CarrotVelocityPathFollower,
    PathFollower,
    PathFollowerConfig,
    StallKicker,
    VelocityFollowerConfig,
    VelocityPathFollower,
)


def test_stall_kicker_hysteresis_survives_noise_spikes():
    # Velocity-estimate noise flickers above the stall threshold while the
    # ball is physically parked. A naive timer resets on every flicker,
    # toggling the kick on/off at a few Hz (visible as board jitter with the
    # ball balanced). The timer must HOLD inside the noise band.
    kicker = StallKicker(kick=0.3, speed_mm_s=8.0, min_duration_s=0.3)

    for i in range(30):  # 0.6s: slow frames with noise spikes into 8..16
        speed = 12.0 if i % 3 == 2 else 2.0  # spike inside the noise band
        kick = kicker.update(speed, 0.02)
    assert kick > 0.0, "noise inside the band must not reset the stall timer"

    # a CLEAR movement (above release threshold) does reset it
    kicker.update(20.0, 0.02)
    assert kicker.update(2.0, 0.02) == 0.0


def test_stall_kicker_escalates_while_stall_persists():
    kicker = StallKicker(kick=0.3, speed_mm_s=8.0, min_duration_s=0.3,
                         ramp_per_s=0.15)
    kick_early, kick_late = 0.0, 0.0
    for i in range(100):  # 2.0s stalled
        k = kicker.update(0.0, 0.02)
        if i == 20:   # 0.42s: just past min duration
            kick_early = k
        if i == 99:   # 2.0s
            kick_late = k
    assert 0.29 < kick_early < 0.35
    assert kick_late > kick_early + 0.15, "kick must escalate during a long stall"


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


def test_velocity_follower_stall_kick_applies_during_corner_crawl():
    cfg = VelocityFollowerConfig(
        v_max_mm_s=30.0,
        min_speed_frac=0.15,
        corner_slow_deg=70.0,
        k_lat=0.0,
        k_vel=0.001,
        max_command=1.0,
        stall_kick=0.5,
        stall_speed_mm_s=8.0,
        stall_request_speed_mm_s=1.0,
        stall_min_duration_s=0.3,
    )
    follower = VelocityPathFollower(cfg)
    path_point = np.array([0.0, 0.0])
    tangent = np.array([1.0, 0.0])

    command = None
    for _ in range(20):
        command, desired = follower.command(
            position_mm=np.array([0.0, 0.0]),
            velocity_mm_s=np.array([0.0, 0.0]),
            path_point_mm=path_point,
            tangent=tangent,
            heading_change_deg=90.0,
            dt_s=0.02,
        )

    assert np.isclose(np.linalg.norm(desired), 4.5)
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


def test_carrot_follower_commands_toward_carrot():
    cfg = CarrotVelocityFollowerConfig(
        v_max_mm_s=20.0, k_vel=0.1, max_command=10.0,
        stall_kick=0.0,
    )
    follower = CarrotVelocityPathFollower(cfg)

    command, desired = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([0.0, 0.0]),
        carrot_mm=np.array([10.0, 0.0]),
        heading_change_deg=0.0,
    )

    assert np.allclose(desired, [20.0, 0.0])
    assert np.allclose(command, [2.0, 0.0])


def test_carrot_follower_brakes_when_ball_is_too_fast():
    cfg = CarrotVelocityFollowerConfig(
        v_max_mm_s=20.0, k_vel=0.1, max_command=10.0,
        stall_kick=0.0,
    )
    follower = CarrotVelocityPathFollower(cfg)

    command, desired = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([30.0, 0.0]),
        carrot_mm=np.array([10.0, 0.0]),
        heading_change_deg=0.0,
    )

    assert np.allclose(desired, [20.0, 0.0])
    assert np.allclose(command, [-1.0, 0.0])


def test_carrot_follower_corner_slowdown_keeps_carrot_direction():
    cfg = CarrotVelocityFollowerConfig(
        v_max_mm_s=20.0, min_speed_frac=0.25, corner_slow_deg=100.0,
        k_vel=0.1, max_command=10.0, stall_kick=0.0,
    )
    follower = CarrotVelocityPathFollower(cfg)

    command, desired = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([0.0, 0.0]),
        carrot_mm=np.array([0.0, 10.0]),
        heading_change_deg=90.0,
    )

    assert np.isclose(np.linalg.norm(desired), 5.0)
    assert np.allclose(desired, [0.0, 5.0])
    assert np.allclose(command, [0.0, 0.5])


def test_carrot_follower_stall_kick_requires_persistence():
    cfg = CarrotVelocityFollowerConfig(
        v_max_mm_s=30.0, k_vel=0.001,
        max_command=1.0, stall_kick=0.5, stall_speed_mm_s=8.0,
        stall_request_speed_mm_s=1.0, stall_min_duration_s=0.3,
    )
    follower = CarrotVelocityPathFollower(cfg)

    command, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([0.0, 0.0]),
        carrot_mm=np.array([10.0, 0.0]),
        heading_change_deg=0.0,
        dt_s=0.02,
    )
    assert abs(command[0]) < 0.05

    for _ in range(20):
        command, _ = follower.command(
            position_mm=np.array([0.0, 0.0]),
            velocity_mm_s=np.array([0.0, 0.0]),
            carrot_mm=np.array([10.0, 0.0]),
            heading_change_deg=0.0,
            dt_s=0.02,
        )
    assert np.isclose(command[0], 0.5)


def test_carrot_follower_reset_clears_persistence_timer():
    cfg = CarrotVelocityFollowerConfig(
        v_max_mm_s=30.0, k_vel=0.001,
        max_command=1.0, stall_kick=0.5, stall_speed_mm_s=8.0,
        stall_request_speed_mm_s=1.0, stall_min_duration_s=0.3,
    )
    follower = CarrotVelocityPathFollower(cfg)

    for _ in range(20):
        follower.command(
            position_mm=np.array([0.0, 0.0]),
            velocity_mm_s=np.array([0.0, 0.0]),
            carrot_mm=np.array([10.0, 0.0]),
            heading_change_deg=0.0,
            dt_s=0.02,
        )
    follower.reset()

    command, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([0.0, 0.0]),
        carrot_mm=np.array([10.0, 0.0]),
        heading_change_deg=0.0,
        dt_s=0.02,
    )
    assert abs(command[0]) < 0.05
