# CPS-ML Maze

Autonomous marble maze project using a camera, a two-axis servo-actuated wooden labyrinth board, and a classical vision/control software stack.

The target system is not a direct CyberRunner clone. It uses the same high-level idea of camera-based state estimation and motorized board control, but the implementation is adapted for this project's hardware: a USB global-shutter camera, Arduino UNO R4 Minima, PCA9685 PWM servo driver, and hobby servos.

## Goal

Solve the full physical maze without user input after manual ball placement/reset.

The first working target is a reliable classical-control solver:

```text
camera -> ball tracking -> maze coordinates -> path planner -> controller -> Arduino -> servos
```

Reinforcement learning is intentionally not required for the initial demo.

## Repository Map

- `src/cps_maze/` - Python package for camera, vision, planning, control, serial hardware interface, and logging.
- `firmware/arduino/maze_servo_controller/` - Arduino firmware for UNO R4 + PCA9685.
- `configs/` - Runtime configuration files.
- `calibration/` - Camera/board calibration files and notes.
- `data/` - Local run data, videos, and processed logs.
- `scripts/` - Operator scripts for camera checks, servo tests, and autonomous runs.
- `docs/` - System context, architecture, hardware notes, validation plan, and AI-agent handoff docs.
- `logs/agent/` - Human/AI development handoff logs.
- `tests/` - Unit tests for pure Python logic.

## Quick Start

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Upload the Arduino firmware from:

```text
firmware/arduino/maze_servo_controller/maze_servo_controller.ino
```

Run initial checks:

```bash
python scripts/check_camera.py --config configs/default.yaml
python scripts/manual_servo_test.py --config configs/default.yaml --neutral
python scripts/servo_sweep_test.py --config configs/default.yaml --axis yaw --amplitude 0.10
```

Create an initial board homography from measured correspondences:

```bash
python scripts/create_homography_from_csv.py \
  --points-csv calibration/example_marker_points.csv \
  --output calibration/board_homography.npz
```

## Current Project Status

The project is at scaffold stage. Before autonomous runs, the team must:

1. Verify servo PWM direction and safe travel limits.
2. Mount servos and camera rigidly.
3. Calibrate camera intrinsics and board homography.
4. Annotate the maze path and holes.
5. Tune the first segment controller.

See [TODO.md](TODO.md) and [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

The initial scaffold validation is documented in [docs/SCAFFOLD_VALIDATION.md](docs/SCAFFOLD_VALIDATION.md).
