#!/usr/bin/env python3
"""Annotate the maze path (and holes) on a camera frame.

Requires a saved homography (run scripts/calibrate_homography.py first).
The maze is static, so this is done once: the clicked centerline encodes all
wall knowledge, and the controller just follows it.

Mouse:
  left click   : add a path waypoint (start -> goal, along the channel centerline)
  right click  : mark a hole (click its center)

Keys:
  SPACE : grab a fresh frame
  u     : undo last waypoint      U : undo last hole
  s     : save path + holes CSVs, then enter verify mode (overlay on live video)
  q/Esc : quit

Output CSVs are in board millimetres:
  path:  x_mm,y_mm            (loadable by WaypointPath.from_csv)
  holes: x_mm,y_mm,radius_mm
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config

WINDOW = "annotate path"


def draw_annotations(
    frame: np.ndarray,
    waypoints_px: list[tuple[int, int]],
    holes_px: list[tuple[int, int]],
    hole_radius_px: int,
) -> np.ndarray:
    out = frame.copy()
    if len(waypoints_px) >= 2:
        cv2.polylines(out, [np.array(waypoints_px, dtype=np.int32)], False, (0, 255, 0), 2)
    for i, (x, y) in enumerate(waypoints_px):
        color = (255, 0, 0) if i == 0 else (0, 255, 0)
        cv2.circle(out, (x, y), 4, color, -1)
    if waypoints_px:
        cv2.putText(out, "start", (waypoints_px[0][0] + 6, waypoints_px[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    for x, y in holes_px:
        cv2.circle(out, (x, y), hole_radius_px, (0, 0, 255), 2)
    cv2.putText(out, f"waypoints: {len(waypoints_px)}  holes: {len(holes_px)}  (s=save)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def save_csvs(
    homography: Homography,
    waypoints_px: list[tuple[int, int]],
    holes_px: list[tuple[int, int]],
    path_file: Path,
    holes_file: Path,
    hole_radius_mm: float,
) -> None:
    path_file.parent.mkdir(parents=True, exist_ok=True)
    with path_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mm", "y_mm"])
        for x_px, y_px in waypoints_px:
            x_mm, y_mm = homography.image_point_to_board_mm(x_px, y_px)
            writer.writerow([f"{x_mm:.1f}", f"{y_mm:.1f}"])
    print(f"saved {len(waypoints_px)} waypoints -> {path_file}")

    if not holes_px:
        print(f"no holes clicked - leaving {holes_file} untouched "
              "(use scripts/auto_detect_holes.py for automatic hole detection)")
        return
    with holes_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mm", "y_mm", "radius_mm"])
        for x_px, y_px in holes_px:
            x_mm, y_mm = homography.image_point_to_board_mm(x_px, y_px)
            writer.writerow([f"{x_mm:.1f}", f"{y_mm:.1f}", f"{hole_radius_mm:.1f}"])
    print(f"saved {len(holes_px)} holes -> {holes_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--path-output", default="configs/maze_path.csv")
    parser.add_argument("--holes-output", default="configs/maze_holes.csv")
    parser.add_argument("--hole-radius-mm", type=float, default=8.0,
                        help="Radius recorded for every clicked hole")
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)

    waypoints_px: list[tuple[int, int]] = []
    holes_px: list[tuple[int, int]] = []
    saved = False

    def on_mouse(event: int, x: int, y: int, *_rest) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            waypoints_px.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            holes_px.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)

    with CameraCapture(config.camera) as camera:
        frame = camera.read().image
        # a representative px radius for hole drawing, from mm via the homography scale
        origin = np.array(homography.board_point_to_image_px(0.0, 0.0))
        offset = np.array(homography.board_point_to_image_px(args.hole_radius_mm, 0.0))
        hole_radius_px = max(int(np.linalg.norm(offset - origin)), 3)

        while True:
            if not saved:
                view = draw_annotations(frame, waypoints_px, holes_px, hole_radius_px)
            else:
                # verify mode: reproject the saved mm path onto live video
                live = camera.read().image
                view = draw_annotations(live, waypoints_px, holes_px, hole_radius_px)
                cv2.putText(view, "VERIFY: path should hug the channel", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(16) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" ") and not saved:
                frame = camera.read().image
                print("grabbed fresh frame")
            elif key == ord("u") and waypoints_px:
                waypoints_px.pop()
            elif key == ord("U") and holes_px:
                holes_px.pop()
            elif key == ord("s"):
                if len(waypoints_px) < 2:
                    print("need at least 2 waypoints before saving")
                    continue
                save_csvs(
                    homography, waypoints_px, holes_px,
                    Path(args.path_output), Path(args.holes_output),
                    args.hole_radius_mm,
                )
                saved = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
