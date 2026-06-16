import numpy as np

from cps_maze.control.pid import PathFollower, PathFollowerConfig


def test_path_follower_clamps_command():
    follower = PathFollower(PathFollowerConfig(kp=10.0, kd=0.0, max_command=1.0))

    command = follower.command(
        position_mm=np.array([0.0, 0.0]),
        velocity_mm_s=np.array([0.0, 0.0]),
        target_mm=np.array([10.0, -10.0]),
    )

    assert np.allclose(command, [1.0, -1.0])

