#!/usr/bin/env python3
"""Closed-loop path following: camera -> ball -> PD controller -> servos.

Prerequisites (each has its own script, test them in order):
  1. calibration/board_homography.npz   (scripts/calibrate_homography.py)
  2. a real path CSV                    (scripts/annotate_path.py)
  3. calibration/axis_map.npz           (scripts/axis_check.py)

First closed-loop test: start with --dry-run (no servos, overlay only), then
run for real with a low cap, e.g. --max-command 0.2, on a short straight
segment before attempting the full maze.

Keys in the preview window: q/Esc = stop (returns to neutral).
"""
from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.control.axis_map import AxisMap
from cps_maze.control.pid import PathFollower, PathFollowerConfig
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand
from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.planning.path import WaypointPath
from cps_maze.vision.ball_tracker import BrightBlobBallTracker
from cps_maze.vision.state_estimator import LowPassVelocityEstimator

WINDOW = "autonomous run"


def load_holes(path: Path) -> np.ndarray:
    """Returns (N, 3) array of x_mm, y_mm, radius_mm; empty if file missing."""
    if not path.exists():
        return np.zeros((0, 3))
    rows = np.genfromtxt(path, delimiter=",", names=True)
    rows = np.atleast_1d(rows)
    return np.column_stack([rows["x_mm"], rows["y_mm"], rows["radius_mm"]]).astype(float)


def draw_overlay(
    image: np.ndarray,
    homography: Homography,
    path: WaypointPath,
    holes: np.ndarray,
    ball_px: tuple[float, float] | None,
    target_mm: np.ndarray | None,
    servo_cmd: np.ndarray,
    status: str,
) -> np.ndarray:
    out = image.copy()
    path_px = homography.board_points_to_image_px(path.points_mm).astype(np.int32)
    cv2.polylines(out, [path_px], False, (0, 255, 0), 2)
    cv2.circle(out, tuple(path_px[-1]), 8, (255, 0, 255), 2)  # goal
    for x_mm, y_mm, r_mm in holes:
        center = homography.board_point_to_image_px(x_mm, y_mm)
        edge = homography.board_point_to_image_px(x_mm + r_mm, y_mm)
        radius = int(np.hypot(edge[0] - center[0], edge[1] - center[1]))
        cv2.circle(out, (int(center[0]), int(center[1])), max(radius, 3), (0, 0, 255), 2)
    if target_mm is not None:
        tx, ty = homography.board_point_to_image_px(float(target_mm[0]), float(target_mm[1]))
        cv2.circle(out, (int(tx), int(ty)), 6, (0, 255, 255), -1)
    if ball_px is not None:
        cv2.circle(out, (int(ball_px[0]), int(ball_px[1])), 6, (255, 0, 0), 2)
    cv2.putText(out, f"yaw={servo_cmd[0]:+.2f} pitch={servo_cmd[1]:+.2f}  {status}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--path", default=None, help="Path CSV override")
    parser.add_argument("--holes", default="configs/maze_holes.csv")
    parser.add_argument("--axis-map", default="calibration/axis_map.npz")
    parser.add_argument("--port", default=None, help="Serial port override, e.g. COM10")
    parser.add_argument("--log", default="data/raw/autonomous_run.csv")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="No serial output; visualize what the controller would do")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--max-command", type=float, default=None,
                        help="Cap |servo command|; start low (e.g. 0.2)")
    parser.add_argument("--lookahead", type=float, default=None, help="Lookahead mm")
    parser.add_argument("--goal-tolerance-mm", type=float, default=15.0)
    parser.add_argument("--lost-timeout-s", type=float, default=2.0,
                        help="Stop if the ball is undetected this long")
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)
    tracker = BrightBlobBallTracker(config.vision)
    path_file = Path(args.path) if args.path else config.resolve_path(config.maze["path_file"])
    path = WaypointPath.from_csv(path_file)
    holes = load_holes(Path(args.holes))

    axis_map_file = Path(args.axis_map)
    if axis_map_file.exists():
        axis_map = AxisMap.load(axis_map_file)
        print(f"axis map loaded from {axis_map_file}:\n{axis_map.matrix}")
    else:
        axis_map = AxisMap.identity()
        print("WARNING: no axis map found - using identity. Run scripts/axis_check.py "
              "first, or the controller may push the ball the wrong way.")

    kp = args.kp if args.kp is not None else float(config.control["kp"])
    kd = args.kd if args.kd is not None else float(config.control["kd"])
    max_command = (args.max_command if args.max_command is not None
                   else float(config.control["max_command"]))
    lookahead_mm = (args.lookahead if args.lookahead is not None
                    else float(config.control["lookahead_mm"]))

    follower = PathFollower(PathFollowerConfig(kp=kp, kd=kd, max_command=max_command))
    estimator = LowPassVelocityEstimator()
    total_length = float(path.cumulative_lengths[-1])

    log_fields = [
        "timestamp_s", "found", "x_mm", "y_mm", "vx_mm_s", "vy_mm_s",
        "target_x_mm", "target_y_mm", "progress_mm",
        "board_cmd_x", "board_cmd_y", "yaw_command", "pitch_command",
    ]

    if args.dry_run:
        serial_ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
        print("DRY RUN: servos disabled, visualization only")
    else:
        serial_ctx = ArduinoServoLink(
            port=args.port or config.serial["port"],
            baudrate=int(config.serial["baudrate"]),
            timeout_s=float(config.serial["timeout_s"]),
        )

    start_time = monotonic()
    last_seen = monotonic()
    outcome = "stopped by user"

    with CameraCapture(config.camera) as camera, serial_ctx as link, \
            CsvRunLogger(Path(args.log), log_fields) as logger:
        if link is not None:
            time.sleep(2.0)  # Arduino reset after port open
            link.neutral()
        try:
            while True:
                if args.max_seconds > 0 and monotonic() - start_time >= args.max_seconds:
                    outcome = "time limit"
                    break

                frame = camera.read()
                detection = tracker.detect(frame.image)
                servo_cmd = np.zeros(2)
                target = None
                ball_px = None
                status = "ball lost"

                if detection.found and detection.x_px is not None and detection.y_px is not None:
                    last_seen = monotonic()
                    ball_px = (detection.x_px, detection.y_px)
                    board_xy = np.array(
                        homography.image_point_to_board_mm(detection.x_px, detection.y_px),
                        dtype=float,
                    )
                    state = estimator.update(board_xy, frame.timestamp_s)
                    progress = path.nearest_progress_mm(board_xy)
                    target = path.point_at_progress_mm(progress + lookahead_mm)

                    board_cmd = follower.command(state.position_mm, state.velocity_mm_s, target)
                    servo_cmd = np.clip(axis_map.apply(board_cmd), -max_command, max_command)
                    status = f"progress {progress:.0f}/{total_length:.0f} mm"

                    near = holes[
                        np.hypot(holes[:, 0] - board_xy[0], holes[:, 1] - board_xy[1])
                        < holes[:, 2] + 10.0
                    ] if len(holes) else []
                    if len(near):
                        status += "  NEAR HOLE"

                    if link is not None:
                        link.send(ServoCommand(yaw=float(servo_cmd[0]),
                                               pitch=float(servo_cmd[1])))
                    logger.write({
                        "timestamp_s": frame.timestamp_s, "found": True,
                        "x_mm": state.position_mm[0], "y_mm": state.position_mm[1],
                        "vx_mm_s": state.velocity_mm_s[0], "vy_mm_s": state.velocity_mm_s[1],
                        "target_x_mm": target[0], "target_y_mm": target[1],
                        "progress_mm": progress,
                        "board_cmd_x": board_cmd[0], "board_cmd_y": board_cmd[1],
                        "yaw_command": servo_cmd[0], "pitch_command": servo_cmd[1],
                    })

                    if progress >= total_length - args.goal_tolerance_mm:
                        outcome = "GOAL REACHED"
                        break
                else:
                    if link is not None:
                        link.neutral()
                    logger.write({
                        "timestamp_s": frame.timestamp_s, "found": False,
                        "x_mm": "", "y_mm": "", "vx_mm_s": "", "vy_mm_s": "",
                        "target_x_mm": "", "target_y_mm": "", "progress_mm": "",
                        "board_cmd_x": "", "board_cmd_y": "",
                        "yaw_command": 0.0, "pitch_command": 0.0,
                    })
                    if monotonic() - last_seen > args.lost_timeout_s:
                        outcome = "ball lost (fell in a hole?)"
                        break

                if not args.no_preview:
                    view = draw_overlay(frame.image, homography, path, holes,
                                        ball_px, target, servo_cmd, status)
                    cv2.imshow(WINDOW, view)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        outcome = "stopped by user"
                        break
        finally:
            if link is not None:
                link.neutral()
            cv2.destroyAllWindows()

    elapsed = monotonic() - start_time
    print(f"\nrun finished: {outcome} after {elapsed:.1f}s  (log: {args.log})")


if __name__ == "__main__":
    main()
