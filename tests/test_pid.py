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


def test_carrot_follower_brake_may_exceed_max_command():
    # A fast ball needs more tilt to stop than to drive: commands opposing
    # the motion may use brake_max_command; driving stays at max_command.
    cfg = CarrotVelocityFollowerConfig(
        v_max_mm_s=20.0, k_vel=0.1, max_command=0.45,
        brake_max_command=1.0, stall_kick=0.0,
    )
    follower = CarrotVelocityPathFollower(cfg)

    # ball racing +x at 100 mm/s, carrot behind the error -> hard brake
    command, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([100.0, 0.0]),
        carrot_mm=np.array([10.0, 0.0]),
        heading_change_deg=0.0,
    )
    assert command[0] < -0.9  # brake beyond the 0.45 driving cap

    # slow ball accelerating forward: capped at the driving limit
    command, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([2.0, 0.0]),
        carrot_mm=np.array([50.0, 0.0]),
        heading_change_deg=0.0,
    )
    assert abs(command[0]) <= 0.45 + 1e-9


def test_carrot_brake_ceiling_shrinks_with_speed():
    # ABS: tilt is force - a brake tilt still applied when the ball stops
    # launches it backward. The brake ceiling must melt away with speed.
    cfg = CarrotVelocityFollowerConfig(
        v_max_mm_s=25.0, k_vel=0.012, max_command=0.9,
        brake_max_command=0.9, brake_cmd_per_mm_s=0.012,
        brake_cmd_floor=0.06, stall_kick=0.0,
    )
    follower = CarrotVelocityPathFollower(cfg)
    carrot = np.array([100.0, 0.0])

    # fast ball overspeeding toward the carrot: strong brake allowed
    cmd_fast, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([70.0, 0.0]),
        carrot_mm=carrot, heading_change_deg=0.0,
    )
    # slow ball, same overspeed sign: brake must be nearly gone
    cmd_slow, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([30.0, 0.0]),
        carrot_mm=carrot, heading_change_deg=0.0,
    )
    assert cmd_fast[0] < 0 and cmd_slow[0] < 0
    fast_limit = 0.06 + 0.012 * 70.0
    slow_limit = 0.06 + 0.012 * 30.0
    assert abs(cmd_fast[0]) <= fast_limit + 1e-9
    assert abs(cmd_slow[0]) <= slow_limit + 1e-9
    assert abs(cmd_slow[0]) < abs(cmd_fast[0])

    # the launch scenario: ball creeping BACKWARD after a hard stop. The
    # counter-command must be gentle (ABS ceiling at low speed), not a slam
    # that starts the oscillation in the other direction.
    cmd_reverse, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([-8.0, 0.0]),
        carrot_mm=carrot, heading_change_deg=0.0,
    )
    assert cmd_reverse[0] > 0  # opposing the reverse creep
    assert abs(cmd_reverse[0]) <= 0.06 + 0.012 * 8.0 + 1e-9


def test_carrot_driving_commands_not_limited_by_abs():
    cfg = CarrotVelocityFollowerConfig(
        v_max_mm_s=25.0, k_vel=0.05, max_command=0.9,
        brake_cmd_per_mm_s=0.012, brake_cmd_floor=0.06, stall_kick=0.0,
    )
    follower = CarrotVelocityPathFollower(cfg)

    # slow ball being accelerated toward the carrot: full driving authority
    cmd, _ = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([2.0, 0.0]),
        carrot_mm=np.array([100.0, 0.0]), heading_change_deg=0.0,
    )
    assert cmd[0] > 0.5  # not squashed to the ABS ceiling


def test_stall_kicker_accumulates_inside_noise_band():
    # A parked ball whose velocity-estimate noise floor sits INSIDE the
    # hysteresis band must still build stall time (observed: 22s stalls at
    # 0.1 commands because the band held the timer at zero forever).
    kicker = StallKicker(kick=0.3, speed_mm_s=5.0, min_duration_s=0.3)

    kick = 0.0
    for _ in range(50):  # 1.0s entirely inside the band (5..10 mm/s)
        kick = kicker.update(7.0, 0.02)
    assert kick > 0.0, "band-resident noise must not suppress the kick forever"
