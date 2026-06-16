# Decisions

## 2026-06-16 - Use Classical Control First

Decision: Build a classical vision/planning/control solver before attempting RL.

Reason: The project deadline requires full-maze solving, the hardware differs from CyberRunner, and a classical loop is easier to validate incrementally.

## 2026-06-16 - Use Custom Python and Arduino Stack

Decision: Start with Python scripts plus Arduino firmware rather than ROS2.

Reason: ROS2 adds process/middleware complexity that is not necessary for a two-servo, one-camera first demo. The architecture leaves room to migrate later if needed.

## 2026-06-16 - Store Calibration Outside Code

Decision: Calibration artifacts live in `calibration/` and runtime knobs live in `configs/`.

Reason: Camera placement, lighting, servo limits, and maze geometry are setup-specific.

