# Validation Plan

## Top-Level Acceptance

The system solves the full maze autonomously from a manual start/reset.

## Test Ladder

1. Servo bench test: commands move predictably and return to neutral.
2. Servo mounted test: full safe range has no binding.
3. Camera test: stable frame capture with fixed exposure.
4. Ball tracking test: annotated video follows the marble reliably.
5. Homography test: known board points map within target error.
6. Single segment control: ball moves along one straight segment.
7. Corner test: ball turns without falling into nearest hole.
8. Full route test: solve complete maze.
9. Repeatability test: measure success rate over repeated attempts.

## Metrics

- Camera frame rate and latency.
- Ball detection rate.
- Position error in millimeters.
- Run completion time.
- Success rate.
- Number and location of failures.
- Max servo command and saturation frequency.

## Validation Rule

Do not tune full-maze behavior until single segment behavior is repeatable.

