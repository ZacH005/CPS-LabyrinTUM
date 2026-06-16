# Scaffold Validation

Validation date: 2026-06-16

## Checks Run

- File tree inspection.
- `python3 -m compileall -q src scripts tests`.
- `python3 -m pytest -q`.

## Result

- Python compilation passed.
- Pytest could not run in the base environment because `pytest` is not installed. Install dev dependencies with `pip install -e ".[dev]"`.

## Top 5 Improvements Found and Applied

1. Path following should project onto path segments, not jump between nearest waypoints.
   - Applied: `WaypointPath` now supports segment projection and progress-based lookahead.

2. Autonomous runs need logs from the start.
   - Applied: `scripts/run_autonomous.py` now writes CSV logs for ball state, target, and commands.

3. Autonomous runs need an operator stop condition.
   - Applied: added `--max-seconds`; `0` keeps running indefinitely.

4. Homography generation needed a clear entry point.
   - Applied: added `scripts/create_homography_from_csv.py` and `calibration/example_marker_points.csv`.

5. Servo calibration needed a safer workflow than arbitrary command entry.
   - Applied: added `scripts/servo_sweep_test.py` with small default amplitude.

Additional firmware review:

- The first firmware draft only ramped when a serial command arrived, which made one-shot servo tests ineffective.
- Applied: firmware now stores target pulses and continuously ramps toward them while the watchdog requests neutral after timeout.

## Remaining Gaps

- No live calibration point picker yet.
- No camera intrinsic calibration script yet.
- No full UI for tuning threshold values.
- Firmware has not been compiled against the actual Arduino library installation.
- No hardware-in-the-loop tests have been run.
