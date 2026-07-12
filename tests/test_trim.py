import numpy as np

from cps_maze.control.axis_map import AxisMap
from cps_maze.control.trim import NeutralTrim, trim_step
from cps_maze.hardware.serial_link import ServoCommand, apply_trim


def test_apply_trim_offsets_and_clamps():
    out = apply_trim(ServoCommand(yaw=0.2, pitch=-0.1), 0.15, -0.05)
    assert np.isclose(out.yaw, 0.35)
    assert np.isclose(out.pitch, -0.15)

    clamped = apply_trim(ServoCommand(yaw=0.9, pitch=-0.95), 0.3, -0.3)
    assert clamped.yaw == 1.0
    assert clamped.pitch == -1.0


def test_neutral_trim_save_load_roundtrip(tmp_path):
    path = tmp_path / "trim.json"
    NeutralTrim(yaw=0.12, pitch=-0.07).save(path, residual_drift_mm_s=1.1)

    loaded = NeutralTrim.load(path)

    assert np.isclose(loaded.yaw, 0.12)
    assert np.isclose(loaded.pitch, -0.07)


def test_load_if_exists_defaults_to_zero(tmp_path):
    trim = NeutralTrim.load_if_exists(tmp_path / "missing.json")
    assert trim.yaw == 0.0 and trim.pitch == 0.0


def test_trim_step_opposes_drift_and_caps():
    identity = AxisMap.identity()

    step = trim_step(np.array([10.0, -5.0]), identity,
                     gain_cmd_per_mm_s=0.004, max_step=1.0)
    assert step[0] < 0 and step[1] > 0  # against the drift

    capped = trim_step(np.array([500.0, 0.0]), identity,
                       gain_cmd_per_mm_s=0.004, max_step=0.05)
    assert np.isclose(np.linalg.norm(capped), 0.05)


def test_trim_step_routes_through_axis_map():
    # servo yaw drives board -y, servo pitch drives board +x (swapped+flip)
    swapped = AxisMap(matrix=np.array([[0.0, -1.0], [1.0, 0.0]]))

    step = trim_step(np.array([8.0, 0.0]), swapped,
                     gain_cmd_per_mm_s=0.004, max_step=1.0)

    # board correction is -x; through this map that is pitch negative
    assert np.isclose(step[0], 0.0, atol=1e-9)
    assert step[1] < 0


def test_drift_nulling_converges_on_simulated_slant():
    # plant: drift velocity proportional to (true board bias + trim),
    # with stiction: no motion below a breakaway tilt
    true_bias = np.array([0.18, -0.11])  # the slanted table, in command units
    drift_per_cmd = 120.0                # mm/s of drift per command unit
    stiction_cmd = 0.02

    identity = AxisMap.identity()
    trim = np.zeros(2)
    still = 0
    for _ in range(20):
        effective = true_bias + trim
        if np.linalg.norm(effective) < stiction_cmd:
            drift = np.zeros(2)
        else:
            drift = drift_per_cmd * effective
        if np.linalg.norm(drift) < 1.5:
            still += 1
            if still >= 2:
                break
            continue
        still = 0
        trim = trim + trim_step(drift, identity,
                                gain_cmd_per_mm_s=0.004, max_step=0.05)

    assert still >= 2, "drift nulling must converge"
    assert np.linalg.norm(true_bias + trim) < stiction_cmd + 0.01
