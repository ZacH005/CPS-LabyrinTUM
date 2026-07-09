# Live MVP Development Roadmap

Single source of truth for getting the physical maze from "wired up" to
reliable full-maze solves. Part 1 records the MVP bring-up stages (now
complete) and what implements them. Part 2 is the active tuning roadmap with
commands. The ground rules apply to both.

The target beyond the MVP is:

> Detect the ball from the fixed camera, follow the traced route through the
> maze, and send safe servo commands until the ball reaches the finish area
> or the run aborts safely - repeatably.

## Required Context To Read First

Before making a plan or editing code, read:

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/ARCHITECTURE.md`
4. `docs/HARDWARE_NOTES.md`
5. This document
6. The specific scripts/modules named in the current stage

Reference camera view: `calibration/CURRENT_FIXED_CAMERA_VIEW.png`
(fixed camera; start near top-right, finish near bottom-left).

## Setup on every machine (once)

```bash
git pull
pip install -e .
```

Create `configs/local.yaml` (gitignored) with YOUR machine's settings; never
put machine-specific values in `default.yaml`:

```yaml
serial:
  port: "COM10"        # Windows lab PC; Mac: /dev/cu.usbmodemXXXX
camera:
  device_index: 0      # the maze camera (OV9281, reports 1280x800)
```

If you see the laptop webcam, the device index is wrong - indices shift after
replugging/reboot; probe 0-4 and pick the one reporting 1280x800.

## Ground Rules

- Do not use RL.
- Do not optimize the full maze before a short segment works.
- Do not widen servo limits because tracking/control is bad (a stalled servo
  burned the power wiring once already).
- Do not use screen recordings for tracker calibration - record natively
  (`scripts/record_camera.py`).
- Do not silently invent measurements. If a stage needs a clicked point,
  video, ROI polygon, device index, or observed behavior, ask the human.
- New homography => re-run the holes + path tools (their CSVs derive from it).
- Camera physically moved => recalibrate homography; path/holes CSVs stay
  valid (they are stored in board-mm).
- After every autonomous run: `python scripts/analyze_run.py` - read it
  BEFORE changing any knob. Change ONE thing per run.
- ChArUco caution: a homography maps one plane. The pattern is only a valid
  ball-plane calibration target when it lies FLAT on the play surface
  (supported: `calibrate_charuco_homography.py --pattern-x-mm/--pattern-y-mm`).
  The corner-click calibration (`calibrate_homography.py`) is the proven
  default; do not block on ChArUco.

---

# Part 1 - MVP Bring-Up (COMPLETE)

| Stage | Goal | Status / implemented by |
|---|---|---|
| 0 | Fixed camera + runtime conventions | Done - convention block in `configs/default.yaml`, `CameraCapture` (DirectShow/MJPG/buffer-1), `local.yaml` overrides |
| 1 | Native live camera recording | Done - `scripts/record_camera.py`, Stage-1 recording in `data/raw/` |
| 2 | Static confusers + maze ROI | Done - `scripts/select_maze_roi.py`, `pipeline.py --calibrate --roi-file`, `calibration/live_confusers.json` |
| 3 | Pipeline tracker extracted for live use | Done - `src/cps_maze/vision/ball_pipeline.py` (+ `PipelineBallTracker` with click-to-seed), selected via `vision.tracker` config |
| 4 | Coordinate frame decision | Done - homography-space (board mm) via corner-click calibration; verified with grid/border overlays |
| 5 | Waypoint tooling | Done - `scripts/annotate_path.py` (manual), `scripts/auto_trace_path.py` (traces the printed line), `scripts/auto_detect_holes.py` |
| 6 | Axis mapping | Tooling done - `scripts/axis_check.py` + `src/cps_maze/control/axis_map.py`. **Measurement must be REDONE** (Part 2, Stage 1): the saved map predates working ball tracking |
| 7 | First closed-loop segment runner | Done - `scripts/run_autonomous.py`: arming phase (click-seed, SPACE), dry-run, command caps, neutral on loss/finish/abort |
| 8 | Run logging + failure diagnosis | Done - CSV log per run + `scripts/analyze_run.py` (detection rate, progress, cross-track stats, stall episodes by path position) |

First closed-loop runs have happened. Observed failure modes: the ball
"balances" short of the target (static friction deadband - addressed by
`stall_kick`/`ki` in the controller) and gets pinned at corners (geometry -
addressed in Part 2, Stage 3).

---

# Part 2 - The Road To A Perfect Run (ACTIVE)

Do the stages in order.

## Stage 1 - Redo the axis map (current one is corrupted)

The saved map was measured while ball tracking was broken. Redo it:

```bash
python scripts/axis_check.py --amplitude 0.4 --max-amplitude 1.0 --pulse-seconds 1.2
```

Per pulse: place the ball in an open area, CLICK THE BALL in the window
(seeds the tracker), then SPACE. Hover the cursor over the ball and over the
worst glare spot; set `vision.min_specular` between the two readings.

**Done when:** the response matrix has one clearly dominant axis per command
and `calibration/axis_map.npz` is saved.

## Stage 2 - One straight section until it is boring

```bash
# sanity check without servos: roll the ball by hand, watch the target dot
python scripts/run_autonomous.py --dry-run

# real runs on a straight stretch (click ball, SPACE to arm)
python scripts/run_autonomous.py
python scripts/analyze_run.py
```

Tune ONE knob per run (CLI: `--kp --kd --ki --stall-kick --lookahead
--max-command`; defaults in `configs/default.yaml`):

| Symptom | Fix |
|---|---|
| oscillates / zigzags around the line | raise `kd` (0.006 -> 0.009) |
| sluggish, lags behind the target dot | raise `kp` slightly (0.015 -> 0.02) |
| parks / "balances" while off-target | raise `stall_kick` by 0.05 (default 0.30) |
| creeps at the very end of segments | raise `ki` slightly (0.004 -> 0.006) |

**Done when:** 5/5 traverses of the straight with p90 cross-track < ~8 mm
(analyze_run prints it).

## Stage 3 - Corners

Corner failures are geometry, not gains: the lookahead target sits across a
wall and the ball gets pinned into it.

1. Test corners with a smaller lookahead: `--lookahead 12`
2. Still failing at one corner? `analyze_run` prints the path-mm where it
   stalls. Re-trace with denser corner waypoints
   (`python scripts/auto_trace_path.py`) or hand-fix just that section
   (`python scripts/annotate_path.py`), swinging wider of the inside wall.

**Done when:** each corner passes 5/5 in isolation.

## Stage 4 - Holes / traps

Where the channel forces a close pass by a hole, bias the annotated
centerline away from the hole edge and re-save. `analyze_run` shows near-miss
locations.

## Stage 5 - Mechanical honesty pass (30 min, big payoff)

- Tighten tie-rod jam nuts and horn screws. Backlash is a control ceiling no
  software removes: slop = the board angle lags the servo = randomly late
  corrections.
- Verify the board is level at neutral
  (`python scripts/manual_servo_test.py --neutral`); the integral term hides
  small bias but eats headroom.

## Stage 6 - Full maze + speed

Only after Stages 1-5 are stable:

```bash
python scripts/run_autonomous.py            # full path
python scripts/analyze_run.py
```

- Raise `--max-command` gradually (0.45 -> 0.6 -> ...) watching for overshoot
  into walls.
- Measure the demo metric: success rate + completion time over 10 consecutive
  runs. That table is the Evaluation section of the report.

**Done when:** the run reaches finish, or fails in a logged, explainable way -
never with uncontrolled servo motion.

---

## Current tuned baseline (configs/default.yaml)

```yaml
control:
  lookahead_mm: 18.0
  kp: 0.015
  kd: 0.006
  ki: 0.004          # integral: un-sticks + absorbs non-level neutral
  integral_limit: 0.25
  stall_kick: 0.30   # min command when stalled off-target (breaks stiction)
  stall_speed_mm_s: 8.0
  stall_dist_mm: 8.0
  max_command: 0.45
```

Runner controls: click ball = seed tracker, SPACE = start, q = stop (board
returns to neutral). The firmware watchdog levels the board within 0.5 s if
anything crashes.

## Prompt Template For A New Chat

```text
Read AGENTS.md and docs/LIVE_MVP_DEVELOPMENT_ROADMAP.md.
We are working on Part 2, Stage <N>: <stage name>.
Do not implement yet. First inspect the relevant files listed in the stage,
then ask for any human inputs required by that stage.
After the human provides those inputs, create a concrete implementation and
validation plan for only this stage.
Then review whether the plan satisfies the "Done when" criteria before
executing.
```
