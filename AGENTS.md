# AI Agent Guide

This file is the first stop for AI coding agents working in this repository.

## Project Intent

Build an autonomous classical-control marble maze solver. The system uses a camera to estimate ball state, a planner/controller to compute board commands, and an Arduino/PCA9685 firmware layer to drive two hobby servos.

Do not assume CyberRunner code is directly compatible. This project uses different actuators, a different maze layout, and a simpler no-RL target.

## Read Before Editing

1. `docs/PROJECT_CONTEXT.md`
2. `docs/ARCHITECTURE.md`
3. `docs/HARDWARE_NOTES.md`
4. `docs/AI_HANDOFF.md`
5. The specific module you plan to edit

## Development Rules

- Keep hardware interfaces narrow and testable.
- Do not put calibration constants directly in code. Use `configs/` or `calibration/`.
- Keep pure logic separate from camera/serial side effects.
- Log enough information to replay failures: timestamps, ball state, target, command, and run state.
- Treat servo limits as safety-critical.
- Prefer small scripts for operator workflows over one large application.

## Current Architecture

```text
scripts/run_autonomous.py
  -> cps_maze.camera
  -> cps_maze.vision.ball_tracker
  -> cps_maze.calibration.homography
  -> cps_maze.planning.path
  -> cps_maze.control.pid
  -> cps_maze.hardware.serial_link
```

## Agent Logging

When making meaningful changes, add a short log in `logs/agent/` using the template in `docs/AI_AGENT_LOG_TEMPLATE.md`.

