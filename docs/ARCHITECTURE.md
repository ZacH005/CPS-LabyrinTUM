# Architecture

## Runtime Data Flow

```text
USB camera
  -> CameraCapture
  -> BrightBlobBallTracker
  -> Homography image_to_board
  -> WaypointPath target selection
  -> PathFollower controller
  -> ArduinoServoLink serial protocol
  -> Arduino UNO R4
  -> PCA9685 PWM driver
  -> yaw/pitch servos
```

## Coordinate Frames

- Image frame: pixels, origin at top-left.
- Board frame: millimeters, origin chosen during calibration.
- Servo command frame: normalized `[-1, 1]` for yaw and pitch.
- PWM frame: microsecond pulses at 50 Hz.

## Process Boundaries

The PC performs camera processing, planning, and control. The Arduino performs deterministic PWM output, command limiting, ramping, and watchdog neutral behavior.

## Serial Protocol

PC to Arduino:

```text
PING
NEUTRAL
SET <yaw_normalized> <pitch_normalized>
```

Arduino to PC:

```text
READY
PONG
OK NEUTRAL
OK SET
ERR <reason>
```

## Safety Boundary

The Arduino is the final safety boundary for servo commands. PC code should clamp commands too, but firmware limits must remain conservative until real mechanical limits are measured.

