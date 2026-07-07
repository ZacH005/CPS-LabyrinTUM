#!/usr/bin/env python3
"""Auto-detect maze holes by thresholding a top-down rectified board view.

Uses the saved homography to warp the camera frame into board-mm space, then
finds dark, circular blobs of hole-like size. Walls (elongated), the printed
guide line (thin), and printed numbers (small) are rejected by area and
circularity filters. Review the result and fix any mistakes by clicking before
saving.

Controls:
  threshold trackbar : adjust until holes are solid dark blobs
  left click         : remove the detection nearest the click
  right click        : add a hole at the click
  SPACE              : grab a fresh frame and re-detect
  s                  : save holes CSV
  q/Esc              : quit
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

WINDOW = "auto detect holes"
SCALE_PX_PER_MM = 2.0


def warp_topdown(
    image: np.ndarray,
    homography: Homography,
    max_out_px: int = 1800,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rectify the camera view into board-mm space, auto-fitting the extent.

    Works with any board frame (corner-click or ChArUco, y-up or y-down,
    offset origins): the output covers wherever the camera image lands in
    board coordinates. Returns (warped, origin_mm, scale_px_per_mm) where
    mm = px / scale + origin_mm.
    """
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    mm = cv2.perspectiveTransform(corners, homography.image_to_board).reshape(-1, 2)
    x_min, y_min = mm.min(axis=0)
    x_max, y_max = mm.max(axis=0)
    span = max(x_max - x_min, y_max - y_min, 1e-6)
    scale = min(SCALE_PX_PER_MM, max_out_px / span)
    shift = np.array([[scale, 0, -x_min * scale],
                      [0, scale, -y_min * scale],
                      [0, 0, 1]])
    size = (int((x_max - x_min) * scale) + 1, int((y_max - y_min) * scale) + 1)
    warped = cv2.warpPerspective(image, shift @ homography.image_to_board, size)
    return warped, np.array([x_min, y_min], dtype=float), float(scale)


def detect_holes(
    topdown_bgr: np.ndarray,
    threshold: int,
    min_radius_mm: float,
    max_radius_mm: float,
    origin_mm: np.ndarray | None = None,
    scale: float = SCALE_PX_PER_MM,
    board_mm: tuple[float, float] | None = None,
    board_margin_mm: float = 5.0,
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Returns (holes as (x_mm, y_mm, r_mm), binary debug mask).

    If board_mm=(width, height) is given, detections outside the board
    rectangle (plus margin) are rejected.
    """
    if origin_mm is None:
        origin_mm = np.zeros(2)
    gray = cv2.cvtColor(topdown_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    # close small gaps (e.g. glare inside a hole) without merging walls
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    min_area = np.pi * (min_radius_mm * scale) ** 2
    max_area = np.pi * (max_radius_mm * scale) ** 2

    holes: list[tuple[float, float, float]] = []
    # RETR_LIST, not RETR_EXTERNAL: at high thresholds the board border forms
    # a closed white ring, and EXTERNAL would drop every hole enclosed by it.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.65:  # walls/bars and line fragments are elongated
            continue
        (x_px, y_px), r_px = cv2.minEnclosingCircle(contour)
        x_mm = x_px / scale + origin_mm[0]
        y_mm = y_px / scale + origin_mm[1]
        if board_mm is not None:
            if not (-board_margin_mm <= x_mm <= board_mm[0] + board_margin_mm
                    and -board_margin_mm <= y_mm <= board_mm[1] + board_margin_mm):
                continue
        # de-duplicate: RETR_LIST can yield outer+inner contours of one blob
        if any(np.hypot(x_mm - hx, y_mm - hy) < max(r_px / scale, hr)
               for hx, hy, hr in holes):
            continue
        holes.append((x_mm, y_mm, r_px / scale))
    return holes, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--output", default="configs/maze_holes.csv")
    parser.add_argument("--min-radius-mm", type=float, default=4.0)
    parser.add_argument("--max-radius-mm", type=float, default=12.0)
    parser.add_argument("--threshold", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)
    try:
        board_mm = (float(config.maze["width_mm"]), float(config.maze["height_mm"]))
    except (KeyError, TypeError, ValueError):
        board_mm = None

    holes: list[tuple[float, float, float]] = []
    manual_added: list[tuple[float, float, float]] = []
    default_r_mm = (args.min_radius_mm + args.max_radius_mm) / 2.0
    frame_info = {"origin": np.zeros(2), "scale": SCALE_PX_PER_MM}

    def on_mouse(event: int, x: int, y: int, *_rest) -> None:
        origin, scale = frame_info["origin"], frame_info["scale"]
        x_mm, y_mm = x / scale + origin[0], y / scale + origin[1]
        frame_info["mouse_mm"] = (x_mm, y_mm)
        if event == cv2.EVENT_LBUTTONDOWN:
            merged = holes + manual_added
            if not merged:
                return
            dists = [np.hypot(h[0] - x_mm, h[1] - y_mm) for h in merged]
            i = int(np.argmin(dists))
            if dists[i] < 20.0:
                target = merged[i]
                (holes if target in holes else manual_added).remove(target)
                print(f"removed hole at ({target[0]:.0f}, {target[1]:.0f}) mm")
        elif event == cv2.EVENT_RBUTTONDOWN:
            manual_added.append((x_mm, y_mm, default_r_mm))
            print(f"added hole at ({x_mm:.0f}, {y_mm:.0f}) mm")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cv2.createTrackbar("threshold", WINDOW, args.threshold, 255, lambda _v: None)
    print(__doc__)

    with CameraCapture(config.camera) as camera:
        frame = camera.read().image
        topdown, origin, scale = warp_topdown(frame, homography)
        frame_info["origin"], frame_info["scale"] = origin, scale
        last_threshold = -1

        def mm_to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
            return (int((x_mm - origin[0]) * scale), int((y_mm - origin[1]) * scale))

        while True:
            threshold = cv2.getTrackbarPos("threshold", WINDOW)
            if threshold != last_threshold:
                holes, mask = detect_holes(topdown, threshold,
                                           args.min_radius_mm, args.max_radius_mm,
                                           origin, scale, board_mm)
                last_threshold = threshold

            view = topdown.copy()
            for x_mm, y_mm, r_mm in holes:
                cv2.circle(view, mm_to_px(x_mm, y_mm), int(r_mm * scale), (0, 0, 255), 2)
            for x_mm, y_mm, r_mm in manual_added:
                cv2.circle(view, mm_to_px(x_mm, y_mm), int(r_mm * scale), (255, 0, 255), 2)
            # frame sanity check: the (0,0)-(w,h) board outline should hug the
            # play area if the active calibration and maze dims are right
            try:
                bw = float(config.maze["width_mm"])
                bh = float(config.maze["height_mm"])
                cv2.rectangle(view, mm_to_px(0.0, 0.0), mm_to_px(bw, bh),
                              (255, 255, 0), 1)
            except (KeyError, TypeError, ValueError):
                pass
            count = len(holes) + len(manual_added)
            mx, my = frame_info.get("mouse_mm", (0.0, 0.0))
            cv2.putText(view, f"holes: {count}  (auto {len(holes)} + manual "
                        f"{len(manual_added)})  s=save", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(view, f"mouse: {mx:.0f}, {my:.0f} mm", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(WINDOW, view)
            cv2.imshow("threshold mask", mask)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                frame = camera.read().image
                topdown, origin, scale = warp_topdown(frame, homography)
                frame_info["origin"], frame_info["scale"] = origin, scale
                last_threshold = -1  # force re-detection
                print("grabbed fresh frame")
            elif key == ord("s"):
                merged = holes + manual_added
                out = Path(args.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["x_mm", "y_mm", "radius_mm"])
                    for x_mm, y_mm, r_mm in merged:
                        writer.writerow([f"{x_mm:.1f}", f"{y_mm:.1f}", f"{r_mm:.1f}"])
                print(f"saved {len(merged)} holes -> {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
