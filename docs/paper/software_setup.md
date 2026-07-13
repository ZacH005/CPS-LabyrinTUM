# Software Stack: What Is Set Up and Fixed

Reference for the paper. Everything below is implemented and validated on the
real rig, and describes the final build that completes the maze end-to-end.
This document is structural: the specific tuning values (controller gains,
speeds, brightness thresholds) live in `configs/default.yaml` and in the
paper's control section, and are not duplicated here.

Repository: https://github.com/ZacH005/CPS-ML-Maze

## 1. System architecture

The system is a classical closed-loop cyber-physical stack. No machine
learning and no reinforcement learning are used; the camera is the only
feedback sensor (the RC servos provide no position feedback).

```
fixed overhead camera (OV9281, global shutter, 1280x800)
  -> ball detection (motion + specular highlight tracker)
  -> homography: image pixels -> board millimetres
  -> path association (windowed, wall-aware)
  -> controller (board-frame command)
  -> axis map (board frame -> servo yaw/pitch)
  -> serial link -> Arduino UNO R4 -> PCA9685 -> 2 servos -> board tilt
```

The control loop runs on the host PC in Python. Each iteration reads the most
recently captured frame from a single-frame buffer (so the loop always acts on
the newest image, never a stale queued one), detects the ball, converts to
board coordinates, finds where it is along the annotated route, computes a tilt
command, and sends it to the Arduino. The loop runs as fast as detection,
overlay drawing, and logging allow -- below the camera's 120 fps capture rate --
so it consumes a subset of captured frames rather than one fixed frame in every
N.

## 2. Coordinate system and calibration

- Board frame: millimetres, origin at the play-area top-left corner, x to
  the right, y downward. Play area measured at 263 mm x 222 mm inside the
  walls.
- The image-to-board mapping is a planar homography stored in
  `calibration/board_homography.npz`. It is calibrated by clicking the four
  play-area corners in a live view (`scripts/calibrate_homography.py`) and
  verified with a reprojected grid overlay. This corner-click calibration is
  the method used throughout; a ChArUco marker-based calibration was
  explored early on and abandoned (see the struggles document).
- All derived artifacts (route, holes, wall mask) are stored in board
  millimetres, so they survive camera moves; only the homography must be
  recalibrated when the camera pose changes. Fixed rule: a new homography
  requires regenerating the derived artifacts.

## 3. One-time board annotation (all implemented and in use)

The maze is static, so board knowledge is captured once per setup:

- Route: `scripts/auto_trace_path.py` automatically traces the printed
  guide line on the board. It isolates thin dark structures by
  morphological thickness separation (the line is thinner than walls),
  orders the points with a gap-jumping greedy trace (the line disappears
  under walls), and simplifies to waypoints. Saved as
  `configs/maze_path_auto.csv` (x_mm, y_mm). A manual click-based
  annotation tool exists as fallback and writes the same format.
- Holes: `scripts/auto_detect_holes.py` finds the holes by thresholding a
  rectified top-down view and filtering blobs by size and circularity, with
  manual click correction. Saved as `configs/maze_holes.csv`
  (x_mm, y_mm, radius_mm).
- Walls: `scripts/build_wall_mask.py` rasterizes the walls once into an
  obstacle mask in board space (`calibration/wall_mask.npz`). Thin printed
  lines are excluded by morphological opening; everything outside the play
  area counts as blocked.

## 4. Perception

Ball detection is the motion + specular-highlight tracker (package module
`cps_maze/vision/ball_pipeline.py`, shared by the offline video pipeline and
all live tools):

- Two complementary cues: frame-to-frame motion (works while the ball
  moves; holes do not move) and near-saturated specular glint (works while
  the ball is stationary; the metal ball glints brighter than holes or
  printed text).
- Static confusers: an offline calibration pass over a recorded video finds
  board locations that are bright suspiciously often (hole rims, glare) and
  permanently excludes them. An ROI polygon excludes everything outside the
  playable surface.
- Track state machine with statuses seed / detected / predicted / lost;
  short gaps are bridged by constant-velocity prediction.
- Seeding policy for demos: click-to-seed. The operator clicks the ball in
  the live window; automatic seeding exists but is not relied on.
- Ball velocity is estimated with a low-pass filtered finite difference,
  guarded against the two failure modes of a naive difference: the velocity
  contribution of frames delivered too close together in time is ignored (a
  near-zero time step would otherwise manufacture an enormous speed), and any
  single measured velocity above a physical ceiling is clamped before it enters
  the filter (a one-frame detection jump is a tracker artifact, not motion).

## 5. Planning and path following (structure)

- The route is a waypoint polyline in board millimetres. The ball's
  position is projected onto it to obtain path progress.
- Association is windowed: the projection may only move within a bounded
  progress window per frame, because the ball cannot teleport along the
  route between frames. This prevents locking onto physically adjacent but
  topologically distant corridors (the maze snakes, so corridors metres
  apart in path order sit millimetres apart behind one wall).
- With the wall mask loaded, candidate projections whose straight line of
  sight from the ball crosses a wall are rejected outright.
- Path curvature ahead of the ball is measured as accumulated absolute
  turning (not endpoint tangent difference, which cancels in chicanes) and
  is used to slow the ball before corners.
- A single speed profile for the whole route is precomputed once at startup.
  It folds corner curvature, committed hole-pass speeds, planned wall
  clearance, per-hole "danger" crawls, and a finish-approach crawl into one
  speed-by-position plan; a backward pass then guarantees every slowdown is
  reachable by braking and a forward pass smooths acceleration. The controller
  looks the plan up slightly ahead of the ball, so slowdowns are anticipated
  rather than discovered late. A separate runtime slowdown from the wall
  distance transform applies only when the ball has drifted off the centerline,
  so the two do not double-count on the route.
- Startup danger check: at load, the route is scanned for holes it passes
  close to at a sharp turn, or nearly grazes on a straight; those spots are
  crawled through under tight control. The finish approach (which threads
  several holes just before the goal) is crawled as one block for the same
  reason.
- Wall-hug detours: for the few holes the traced route grazes, the lookahead
  point the controller chases is shifted a few millimetres perpendicular to
  the path, so the ball rides along the outer wall around the hole and rejoins
  the route after it instead of oscillating into the capture zone. This is a
  planning-time offset on the pursued point, not a change to the stored route.
- An A* replanner that reroutes the ball around obstacles when it drifts far
  off route exists in the codebase but is disabled by default: the overhead
  camera's oblique view of the walls produces false wall contacts that made it
  fire spuriously, and ordinary following plus the off-route wall slowdown
  cover recovery without it (see the struggles document).

## 6. Control and safety (structure)

- The controller works in the board frame; a 2x2 axis map
  (`calibration/axis_map.npz`) converts board-frame commands to servo
  yaw/pitch, absorbing any channel swaps or sign flips in the physical build.
  A calibration script (`scripts/axis_check.py`) pulses each servo axis and
  measures the ball's response to produce this map, but on this rig the
  measurement was unreliable (backlash-skewed and asymmetric), so the final
  build overrides it with an identity matrix and levels the board by hand
  (see the struggles document).
- The default and final controller is carrot following: it chases the
  furthest lookahead point on the route still in unobstructed line of sight,
  and controls the ball's VELOCITY toward that point rather than its position.
  Board tilt commands acceleration, so a velocity loop brakes automatically
  when the ball carries too much speed into a corner, which a position loop
  cannot.
- The velocity loop is a PI controller on speed error. The proportional term
  tracks the planned speed; the integral term is the explicit stiction
  compensator. Because the ball is a static-friction plant, a pure proportional
  command falls to zero exactly when the speed error does and the ball stalls,
  so the integral ramps the tilt up smoothly until the ball breaks free and
  then unwinds as the error closes; an anti-windup bound caps the integral's
  contribution so a long stall cannot wind up into a launch. This continuous
  compensator replaced an earlier pile of discrete "band-aids" -- a breakaway
  stall kick, a displacement-based unstick push, and a composure hold-and-damp
  state -- all of which remain in the codebase but are disabled in the final
  build (see the struggles document).
- Command authority is asymmetric: driving tilt is held to a gentle cap and
  slew rate for smooth motion, while a command that opposes the ball's motion
  (a brake) may use more authority and reverse faster, under a
  speed-proportional ceiling that melts the brake to flat as the ball stops
  (otherwise a residual brake tilt launches a stopped ball backward).
- Two small end-game aids run near the final hole: a one-sided wall-lean push
  that leans the ball against the outer wall (fading out once the ball already
  moves into the wall), and, once the goal is reached, a damped hold that keeps
  the ball settled in the goal pocket until the operator ends the run rather
  than snapping straight to neutral.
- The final build's safety stance is prevention over reaction: the reactive
  last-resort mechanisms (the emergency hole-brake and the composure state)
  are present in the code but disabled, because smooth speed planning, crawls,
  and wall-hug detours keep the ball out of trouble, and the violent reactions
  did more harm than good. The always-on safety layers are:
  - commands are clamped to a configurable cap and rate-limited (slew) before
    reaching hardware;
  - runs start through an arming phase (operator clicks the ball, then
    explicitly starts; the run cannot begin without a tracked ball);
  - the board is commanded to neutral whenever the ball is not confidently
    detected, on timeout, and on operator abort;
  - a slow ball drifting within a few millimetres of a wall gets a small
    corrective push away from it;
  - a startup consistency check disables the wall mask for the run if it reads
    as stale against the current route (more than ~2% of the centerline falls
    inside a wall), since a wrong mask is worse than none;
  - independently of the host, the firmware returns both axes to neutral
    within 500 ms if commands stop arriving (watchdog), clamps pulse widths to
    a safe range, and ramps rather than steps between targets.

## 7. Hardware interface and firmware (fixed)

- Host to Arduino: USB serial at 500000 baud, line-oriented ASCII protocol:
  `SET <yaw> <pitch>` with normalized values in [-1, 1], `NEUTRAL`, and
  `PING`/`PONG`. The firmware maps normalized commands to servo pulse
  widths around a 1500 us neutral.
- Firmware: non-blocking serial reader, 200 Hz servo update schedule,
  I2C at 400 kHz to a PCA9685 PWM driver (address 0x40, 50 Hz servo frame),
  channel 0 and channel 1 driving the two tilt axes.
- Firmware safety (independent of all host software): hard pulse-width
  clamp, slew-rate ramping between targets, 500 ms neutral watchdog.

## 8. Run instrumentation and evaluation method

- Every autonomous run writes a CSV log: timestamp, detection flag, ball
  position and velocity (mm, mm/s), path progress, target, board-frame
  command, and the final servo commands.
- `scripts/analyze_run.py` computes the evaluation metrics from a log:
  detection rate, share of the route reached, ball speed statistics,
  cross-track error statistics (median / p90 / max distance from the route
  centerline), and stall episodes located by path position. These are the
  quantitative metrics used for tuning and will be the basis of the
  evaluation section.

## 9. Software engineering setup

- Python package `cps_maze` under `src/` (camera, vision, calibration,
  planning, control, hardware, logging), with operator scripts under
  `scripts/` and a pytest suite under `tests/` covering the pure logic
  (controllers, path association, curvature, calibration mappings, wall
  map, tracker behavior).
- Camera capture is cross-platform. On Windows the Media Foundation backend is
  used, which is the only backend that negotiates the camera's native high-rate
  mode and delivers the full 120 fps at 1280x800 (the default DirectShow
  backend exposed only an uncompressed mode capped near 10 fps). Its slow
  one-time open is amortized by an optional camera-server process that opens the
  device once and publishes frames into shared memory, so every calibration and
  test tool attaches instantly and reads frames stamped with their true capture
  time; tools fall back to opening the device directly when no server is
  running.
- Machine-specific settings (serial port, camera device index) live in a
  gitignored `configs/local.yaml` overlay; shared configuration lives in
  `configs/default.yaml`.
- Development followed a staged bring-up: electronics smoke test, servo
  direction checks, calibration, perception, teleoperation, then closed
  loop, with each stage validated on hardware before the next. Manual
  teleoperation tools (keyboard and touchpad) exist for testing and for
  driving the ball during calibration recordings.
