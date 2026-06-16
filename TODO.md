# TODO

## Phase 1 - Hardware Bring-Up

- Verify each servo's real PWM range on the bench.
- Confirm whether `0.5 ms`, `1.5 ms`, and `2.5 ms` match the vendor angle claims.
- Choose two servos for yaw and pitch.
- Add a large capacitor across PCA9685 servo power input.
- Add a physical power switch or emergency stop for servo power.
- Mount servos to the wooden maze frame.
- Connect servos to the original board control mechanism.
- Measure safe command limits before firmware limits are widened.
- Verify the board returns to neutral repeatably.

## Phase 2 - Camera and Calibration

- Confirm camera modes with `v4l2-ctl --list-formats-ext`.
- Choose resolution/FPS with stable low latency.
- Lock camera exposure, gain, and focus if supported.
- Add fixed lighting.
- Capture calibration images.
- Estimate camera intrinsics.
- Estimate board homography from image pixels to board coordinates.
- Save calibration outputs in `calibration/`.

## Phase 3 - Vision

- Select marble/background detection strategy for reflective silver ball.
- Implement ball detector tuning UI or config parameters.
- Add ball velocity estimation.
- Add lost-ball and hole-fall detection.
- Record annotated debug videos.

## Phase 4 - Planning and Control

- Take a top-down reference image of the maze.
- Annotate start, goal, holes, and path centerline.
- Implement waypoint/path-following controller.
- Tune one straight section.
- Tune corners.
- Add command saturation and jerk limits.
- Attempt full-maze run.

## Phase 5 - Demo Readiness

- Record repeated full-maze attempts.
- Report success rate and completion time.
- Document setup procedure.
- Create final presentation diagrams and run videos.

