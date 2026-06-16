import numpy as np

from cps_maze.vision.state_estimator import LowPassVelocityEstimator


def test_velocity_estimator_computes_smoothed_velocity():
    estimator = LowPassVelocityEstimator(alpha=1.0)

    estimator.update(np.array([0.0, 0.0]), timestamp_s=0.0)
    state = estimator.update(np.array([10.0, 0.0]), timestamp_s=2.0)

    assert np.allclose(state.velocity_mm_s, [5.0, 0.0])

