#!/usr/bin/env python3
from __future__ import annotations

import argparse

import cv2

from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.vision.ball_tracker import BrightBlobBallTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tracker = BrightBlobBallTracker(config.vision)

    with CameraCapture(config.camera) as camera:
        while True:
            frame = camera.read()
            detection = tracker.detect(frame.image)
            output = tracker.draw_detection(frame.image, detection)
            cv2.imshow("cps-maze camera", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

