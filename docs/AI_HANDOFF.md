# AI Handoff

## Current State

This repo is an initial scaffold. No physical calibration data has been collected yet. The firmware and Python scripts define the intended interfaces but still need hardware validation.

## Next Best Work

1. Upload firmware and run `scripts/manual_servo_test.py`.
2. Confirm real servo PWM direction and range.
3. Update `configs/default.yaml` with safe limits.
4. Run `scripts/check_camera.py`.
5. Tune `vision` thresholds for the reflective silver marble.
6. Add calibration scripts once the camera mount is fixed.
7. Review `docs/SCAFFOLD_VALIDATION.md` before extending the scaffold.

## Important Assumptions

- Only two servos are needed, one for yaw and one for pitch.
- Manual ball reset is acceptable.
- Path can initially be manually annotated as waypoints.
- Classical control is the primary approach.

## Do Not Do Yet

- Do not add RL before the classical loop works.
- Do not widen servo PWM limits without measured mechanical clearance.
- Do not hardcode calibration values into Python modules.
- Do not treat the example path as real maze geometry.
