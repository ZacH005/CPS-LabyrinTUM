#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from time import monotonic

import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.control.pid import PathFollower, PathFollowerConfig
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand
from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.planning.path import WaypointPath
from cps_maze.vision.ball_tracker import BrightBlobBallTracker
from cps_maze.vision.state_estimator import LowPassVelocityEstimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", required=True)
    parser.add_argument("--log", default="data/raw/autonomous_run.csv")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)
    tracker = BrightBlobBallTracker(config.vision)
    path = WaypointPath.from_csv(config.resolve_path(config.maze["path_file"]))
    follower = PathFollower(
        PathFollowerConfig(
            kp=float(config.control["kp"]),
            kd=float(config.control["kd"]),
            max_command=float(config.control["max_command"]),
        )
    )
    estimator = LowPassVelocityEstimator()
    log_fields = [
        "timestamp_s",
        "found",
        "x_mm",
        "y_mm",
        "vx_mm_s",
        "vy_mm_s",
        "target_x_mm",
        "target_y_mm",
        "yaw_command",
        "pitch_command",
    ]
    start_time = monotonic()

    with CameraCapture(config.camera) as camera, ArduinoServoLink(
        port=config.serial["port"],
        baudrate=int(config.serial["baudrate"]),
        timeout_s=float(config.serial["timeout_s"]),
    ) as link, CsvRunLogger(Path(args.log), log_fields) as logger:
        try:
            while True:
                if args.max_seconds > 0 and monotonic() - start_time >= args.max_seconds:
                    break

                frame = camera.read()
                detection = tracker.detect(frame.image)
                if not detection.found or detection.x_px is None or detection.y_px is None:
                    link.neutral()
                    logger.write(
                        {
                            "timestamp_s": frame.timestamp_s,
                            "found": False,
                            "x_mm": "",
                            "y_mm": "",
                            "vx_mm_s": "",
                            "vy_mm_s": "",
                            "target_x_mm": "",
                            "target_y_mm": "",
                            "yaw_command": 0.0,
                            "pitch_command": 0.0,
                        }
                    )
                    continue

                board_xy = np.array(
                    homography.image_point_to_board_mm(detection.x_px, detection.y_px),
                    dtype=float,
                )
                state = estimator.update(board_xy, frame.timestamp_s)

                target = path.target_ahead(board_xy, float(config.control["lookahead_mm"]))
                command = follower.command(state.position_mm, state.velocity_mm_s, target)
                link.send(ServoCommand(yaw=float(command[0]), pitch=float(command[1])))
                logger.write(
                    {
                        "timestamp_s": frame.timestamp_s,
                        "found": True,
                        "x_mm": state.position_mm[0],
                        "y_mm": state.position_mm[1],
                        "vx_mm_s": state.velocity_mm_s[0],
                        "vy_mm_s": state.velocity_mm_s[1],
                        "target_x_mm": target[0],
                        "target_y_mm": target[1],
                        "yaw_command": command[0],
                        "pitch_command": command[1],
                    }
                )
        finally:
                link.neutral()


if __name__ == "__main__":
    main()
