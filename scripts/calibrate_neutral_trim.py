#!/usr/bin/env python3
"""Find the neutral trim: the command at which the board is actually LEVEL.

The table/frame is slanted, so servo neutral (command 0,0) is not a level
board and a resting ball drifts. This tool uses the ball itself as the level
sensor, in two phases:

Phase 1 - drift nulling: stream the current trim, measure the ball's drift
velocity over a window, step the trim against the drift, repeat until the
ball stays still twice in a row.

Phase 2 - bias centering (--no-refine to skip): static friction means there
is a whole BAND of trims where the ball does not move even though the board
is not truly level. For each axis, probe outward until drift starts on both
sides and center the trim in the band - the "perfect zero" with no hidden
bias for the controller to fight.

The result is saved to calibration/neutral_trim.json and applied by the
serial link in every tool automatically, so command (0,0) means "level".

Procedure: place the ball in an OPEN area (no walls/holes nearby - the
middle of the largest corridor), click it, press SPACE. If the ball wanders
too far during calibration the tool pauses and asks you to re-place it.

Keys: click = seed tracker, SPACE = continue, q/Esc = abort.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.control.axis_map import AxisMap
from cps_maze.control.trim import DEFAULT_TRIM_FILE, NeutralTrim, trim_step
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand
from cps_maze.vision.ball_pipeline import make_tracker

WINDOW = "neutral trim calibration"


class MeasurementError(RuntimeError):
    pass


def show(view, lines):
    for k, text in enumerate(lines):
        cv2.putText(view, text, (10, 28 + 26 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.imshow(WINDOW, view)


def wait_for_placement(camera, tracker, link, mouse, prompt) -> np.ndarray | None:
    """Live view; click seeds the tracker; SPACE continues once the ball is
    tracked. Streams the current trim so the board holds its test attitude.
    Returns the ball position in px, or None on abort."""
    while True:
        link.send(ServoCommand(0.0, 0.0))  # pure trim via the link
        frame = camera.read()
        seed = mouse.pop("seed", None)
        if seed is not None and hasattr(tracker, "seed"):
            tracker.seed(*seed)
        det = tracker.detect(frame.image)
        view = tracker.draw_detection(frame.image, det)
        state = "ball locked - SPACE to continue" if det.found else "CLICK THE BALL"
        show(view, [prompt, state,
                    f"trim now: yaw={link.trim_yaw:+.3f} pitch={link.trim_pitch:+.3f}"])
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" ") and det.found:
            return np.array([det.x_px, det.y_px])
        if key in (27, ord("q")):
            return None


def measure_drift(link, camera, tracker, homography, seconds, phase_text,
                  min_samples=8):
    """Stream the current trim while fitting the ball's velocity (mm/s)."""
    times, pts = [], []
    t_end = monotonic() + seconds
    while monotonic() < t_end:
        link.send(ServoCommand(0.0, 0.0))
        frame = camera.read()
        det = tracker.detect(frame.image)
        view = tracker.draw_detection(frame.image, det)
        show(view, [phase_text,
                    f"trim: yaw={link.trim_yaw:+.3f} pitch={link.trim_pitch:+.3f}",
                    "measuring drift..."])
        cv2.waitKey(1)
        if det.found and det.x_px is not None:
            times.append(frame.timestamp_s)
            pts.append(homography.image_point_to_board_mm(det.x_px, det.y_px))
    if len(pts) < min_samples:
        raise MeasurementError(f"only {len(pts)} tracked frames in the window")
    t = np.asarray(times) - times[0]
    p = np.asarray(pts)
    vx = float(np.polyfit(t, p[:, 0], 1)[0])
    vy = float(np.polyfit(t, p[:, 1], 1)[0])
    return np.array([vx, vy]), p[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--axis-map", default="calibration/axis_map.npz")
    parser.add_argument("--port", default=None)
    parser.add_argument("--output", default=DEFAULT_TRIM_FILE)
    parser.add_argument("--window-s", type=float, default=1.5,
                        help="drift measurement window")
    parser.add_argument("--settle-s", type=float, default=0.8)
    parser.add_argument("--still-mm-s", type=float, default=1.5,
                        help="drift below this counts as still")
    parser.add_argument("--onset-mm-s", type=float, default=3.0,
                        help="drift above this marks a band edge in phase 2")
    parser.add_argument("--gain", type=float, default=0.004,
                        help="trim command per mm/s of drift")
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--max-trim", type=float, default=0.35)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--probe-step", type=float, default=0.02)
    parser.add_argument("--max-probes", type=int, default=8)
    parser.add_argument("--no-refine", action="store_true",
                        help="skip phase 2 band centering")
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)
    tracker = make_tracker(config.vision)
    axis_map = (AxisMap.load(args.axis_map) if Path(args.axis_map).exists()
                else AxisMap.identity())
    start = NeutralTrim.load_if_exists(args.output)
    port = args.port or config.serial["port"]

    mouse: dict = {}

    def on_mouse(event, x, y, *_rest):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["seed"] = (x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)

    with CameraCapture(config.camera) as camera, ArduinoServoLink(
        port=port, baudrate=int(config.serial["baudrate"]),
        timeout_s=float(config.serial["timeout_s"]),
        trim_yaw=start.yaw, trim_pitch=start.pitch,
    ) as link:
        time.sleep(2.0)  # Arduino reset
        trim = np.array([start.yaw, start.pitch])
        if start.yaw or start.pitch:
            print(f"starting from existing trim yaw={start.yaw:+.3f} "
                  f"pitch={start.pitch:+.3f}")

        if wait_for_placement(camera, tracker, link, mouse,
                              "PHASE 1: place ball in an OPEN area") is None:
            print("aborted")
            return

        # ---- phase 1: drift nulling ----
        still = 0
        for it in range(args.max_iters):
            time.sleep(args.settle_s)
            try:
                vel, pos_mm = measure_drift(
                    link, camera, tracker, homography, args.window_s,
                    f"PHASE 1  iteration {it + 1}/{args.max_iters}")
            except MeasurementError as exc:
                print(f"measurement failed ({exc}); re-place the ball")
                if wait_for_placement(camera, tracker, link, mouse,
                                      "re-place the ball") is None:
                    return
                continue
            speed = float(np.linalg.norm(vel))
            print(f"[{it + 1:2d}] trim=({trim[0]:+.3f},{trim[1]:+.3f})  "
                  f"drift {speed:5.1f} mm/s  ({vel[0]:+.1f},{vel[1]:+.1f})")
            if speed < args.still_mm_s:
                still += 1
                if still >= 2:
                    print("ball is still - phase 1 converged")
                    break
                continue
            still = 0
            delta = trim_step(vel, axis_map, args.gain, args.max_step)
            trim = np.clip(trim + delta, -args.max_trim, args.max_trim)
            link.set_trim(*trim)
        else:
            print("WARNING: phase 1 did not fully converge; saving best effort")

        # ---- phase 2: center the stiction band ("the perfect zero") ----
        if not args.no_refine:
            for axis, name in ((0, "yaw"), (1, "pitch")):
                if wait_for_placement(
                        camera, tracker, link, mouse,
                        f"PHASE 2 ({name}): re-center ball in the open area"
                ) is None:
                    return
                edges = []
                for sign in (+1.0, -1.0):
                    edge = trim[axis]
                    for k in range(1, args.max_probes + 1):
                        probe = trim.copy()
                        probe[axis] = trim[axis] + sign * k * args.probe_step
                        if abs(probe[axis]) > args.max_trim:
                            break
                        link.set_trim(*probe)
                        time.sleep(args.settle_s)
                        try:
                            vel, _ = measure_drift(
                                link, camera, tracker, homography,
                                max(args.window_s * 0.8, 1.0),
                                f"PHASE 2  {name} edge probe "
                                f"{'+' if sign > 0 else '-'}{k}")
                        except MeasurementError:
                            break
                        if float(np.linalg.norm(vel)) > args.onset_mm_s:
                            # drift started: edge is just inside this probe
                            edge = probe[axis] - sign * args.probe_step * 0.5
                            break
                        edge = probe[axis]
                    edges.append(edge)
                    link.set_trim(*trim)
                    time.sleep(args.settle_s)
                center = (edges[0] + edges[1]) / 2.0
                print(f"{name}: stiction band [{min(edges):+.3f}, "
                      f"{max(edges):+.3f}] -> centered at {center:+.3f}")
                trim[axis] = center
                link.set_trim(*trim)

        # ---- final verification ----
        if wait_for_placement(camera, tracker, link, mouse,
                              "VERIFY: re-center the ball one last time") is None:
            return
        time.sleep(args.settle_s)
        try:
            vel, _ = measure_drift(link, camera, tracker, homography,
                                   args.window_s * 1.5, "VERIFY")
            residual = float(np.linalg.norm(vel))
        except MeasurementError:
            residual = float("nan")
        link.neutral()

    cv2.destroyAllWindows()
    NeutralTrim(yaw=float(trim[0]), pitch=float(trim[1])).save(
        args.output, residual_drift_mm_s=residual)
    print(f"\nsaved neutral trim -> {args.output}")
    print(f"trim: yaw={trim[0]:+.3f} pitch={trim[1]:+.3f}  "
          f"residual drift {residual:.1f} mm/s")
    print("All tools apply this automatically via the serial link: "
          "command (0,0) now means LEVEL.")


if __name__ == "__main__":
    main()
