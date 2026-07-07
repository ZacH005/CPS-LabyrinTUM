#!/usr/bin/env python3
"""Interactive board homography calibration.

Click the four corners of the maze PLAY AREA (the surface the ball rolls on,
inside the walls) in this order:

    1. top-left    2. top-right    3. bottom-right    4. bottom-left

"Top-left" here defines the board coordinate origin (0, 0); x grows to the
right (toward corner 2) and y grows downward (toward corner 4). The physical
board width/height in mm are taken from the config (maze.width_mm/height_mm).

Keys:
  SPACE : grab a fresh frame (do this after nudging the camera)
  u     : undo last click
  s     : solve + save homography, then enter verify mode
  q/Esc : quit

Verify mode shows a reprojected mm grid over live video and prints the board
position under the mouse. The grid must hug the play area edges; if it drifts,
recalibrate.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography, estimate_homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config

CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]
WINDOW = "calibrate homography"


def draw_state(frame: np.ndarray, clicks: list[tuple[int, int]]) -> np.ndarray:
    out = frame.copy()
    for i, (x, y) in enumerate(clicks):
        cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(out, CORNER_NAMES[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if len(clicks) >= 2:
        cv2.polylines(out, [np.array(clicks, dtype=np.int32)], len(clicks) == 4, (0, 255, 255), 1)
    prompt = ("all corners set - press s to solve/save"
              if len(clicks) == 4 else f"click {CORNER_NAMES[len(clicks)]}")
    cv2.putText(out, prompt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def draw_grid(frame: np.ndarray, homography: Homography, width_mm: float,
              height_mm: float, step_mm: float = 50.0) -> np.ndarray:
    out = frame.copy()
    for x in np.arange(0.0, width_mm + 1e-6, step_mm):
        line = np.array([[x, 0.0], [x, height_mm]])
        p = homography.board_points_to_image_px(line).astype(int)
        cv2.line(out, tuple(p[0]), tuple(p[1]), (0, 255, 255), 1)
    for y in np.arange(0.0, height_mm + 1e-6, step_mm):
        line = np.array([[0.0, y], [width_mm, y]])
        p = homography.board_points_to_image_px(line).astype(int)
        cv2.line(out, tuple(p[0]), tuple(p[1]), (0, 255, 255), 1)
    border = np.array([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]])
    p = homography.board_points_to_image_px(border).astype(int)
    cv2.polylines(out, [p], True, (0, 0, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="calibration/board_homography.npz")
    args = parser.parse_args()

    config = load_config(args.config)
    width_mm = float(config.maze["width_mm"])
    height_mm = float(config.maze["height_mm"])
    board_corners_mm = np.array(
        [[0.0, 0.0], [width_mm, 0.0], [width_mm, height_mm], [0.0, height_mm]]
    )

    clicks: list[tuple[int, int]] = []
    mouse_px = [0, 0]

    def on_mouse(event: int, x: int, y: int, *_rest) -> None:
        mouse_px[0], mouse_px[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            print(f"corner {CORNER_NAMES[len(clicks) - 1]}: ({x}, {y})")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    homography: Homography | None = None
    print(__doc__)

    with CameraCapture(config.camera) as camera:
        frame = camera.read().image
        while True:
            if homography is None:
                view = draw_state(frame, clicks)
            else:
                frame = camera.read().image  # verify mode is live
                view = draw_grid(frame, homography, width_mm, height_mm)
                x_mm, y_mm = homography.image_point_to_board_mm(*mouse_px)
                cv2.putText(view, f"mouse: {x_mm:.1f}, {y_mm:.1f} mm", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(16) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" ") and homography is None:
                frame = camera.read().image
                print("grabbed fresh frame")
            elif key == ord("u") and clicks:
                clicks.pop()
            elif key == ord("s") and len(clicks) == 4:
                image_points = np.array(clicks, dtype=float)
                homography = estimate_homography(image_points, board_corners_mm)
                homography.save(args.output)
                # report reprojection error at the four corners
                reproj = homography.board_points_to_image_px(board_corners_mm)
                err = np.linalg.norm(reproj - image_points, axis=1)
                print(f"saved {args.output}")
                print(f"corner reprojection error px: {err.round(2)} (max {err.max():.2f})")
                print("verify mode: grid should hug the play area; q to quit")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
