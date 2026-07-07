#!/usr/bin/env python3
"""DEMO: auto-trace the printed guide line into a path CSV.

Approach: the printed line is thinner than the walls, so a morphological
opening (kernel wider than the line, narrower than a wall) erases the line
but keeps walls/holes; subtracting isolates thin structures. Small components
(printed numbers) are dropped, then the line is traced start->finish with a
greedy nearest-neighbour walk that can jump small gaps (where the line passes
under a wall), preferring candidates that continue the current heading. The
trace is simplified to sparse waypoints.

This is a best-effort demo: if the trace goes wrong (shortcuts between
adjacent corridors, big gaps), fall back to scripts/annotate_path.py - the
output format is identical.

Controls:
  threshold trackbar : same as hole detection
  left click         : seed the trace at the START of the line, tracing begins
  r                  : reset trace
  SPACE              : fresh frame
  s                  : save traced path CSV
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

from auto_detect_holes import SCALE_PX_PER_MM, warp_topdown

WINDOW = "auto trace path"


def extract_line_mask(
    topdown_bgr: np.ndarray,
    threshold: int,
    max_line_width_mm: float,
    min_component_mm: float,
    scale: float = SCALE_PX_PER_MM,
) -> np.ndarray:
    """Isolate thin dark structures (the printed line) from walls/holes/text."""
    gray = cv2.cvtColor(topdown_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    # opening with a kernel wider than the line erases it; walls survive
    k = max(int(max_line_width_mm * scale) | 1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    thin = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.dilate(thick, None)))

    # drop small components: printed numbers, specks, wall fringes
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thin)
    keep = np.zeros_like(thin)
    min_diag_px = min_component_mm * scale
    for i in range(1, n):
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if np.hypot(w, h) >= min_diag_px:
            keep[labels == i] = 255
    return keep


def line_points_mm(
    line_mask: np.ndarray,
    grid_mm: float = 1.5,
    origin_mm: np.ndarray | None = None,
    scale: float = SCALE_PX_PER_MM,
) -> np.ndarray:
    """Subsample line pixels to a coarse grid of points in mm, shape (N, 2)."""
    if origin_mm is None:
        origin_mm = np.zeros(2)
    ys, xs = np.nonzero(line_mask)
    pts = np.column_stack([xs, ys]) / scale + origin_mm
    quantized = np.round(pts / grid_mm).astype(int)
    _, idx = np.unique(quantized, axis=0, return_index=True)
    return pts[idx]


def trace_line(
    points: np.ndarray,
    seed_mm: np.ndarray,
    step_mm: float = 4.0,
    gap_mm: float = 18.0,
    visit_radius_mm: float = 3.0,
) -> np.ndarray:
    """Greedy nearest-neighbour walk with heading-biased gap jumps."""
    remaining = points.copy()
    current = remaining[np.argmin(np.linalg.norm(remaining - seed_mm, axis=1))]
    heading = None
    trace = [current]

    def consume(center: np.ndarray, radius: float) -> None:
        nonlocal remaining
        d = np.linalg.norm(remaining - center, axis=1)
        remaining = remaining[d > radius]

    consume(current, visit_radius_mm)
    while len(remaining):
        d = np.linalg.norm(remaining - current, axis=1)
        near = d <= step_mm
        if not near.any():
            near = d <= gap_mm  # gap jump (line passes under a wall)
            if not near.any():
                break
        candidates = remaining[near]
        dists = d[near]
        if heading is None:
            i = int(np.argmin(dists))
        else:
            vecs = candidates - current
            norms = np.maximum(np.linalg.norm(vecs, axis=1), 1e-9)
            cos = (vecs @ heading) / norms
            # prefer close candidates that keep the current direction
            score = dists * (1.6 - cos)
            i = int(np.argmin(score))
        nxt = candidates[i]
        step = nxt - current
        norm = np.linalg.norm(step)
        if norm > 1e-9:
            new_heading = step / norm
            heading = new_heading if heading is None else (
                0.7 * new_heading + 0.3 * heading
            )
            heading /= np.linalg.norm(heading)
        current = nxt
        trace.append(current)
        consume(current, visit_radius_mm)
    return np.array(trace)


def simplify(trace_mm: np.ndarray, epsilon_mm: float = 2.0) -> np.ndarray:
    if len(trace_mm) < 3:
        return trace_mm
    pts = trace_mm.astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts, epsilon_mm, False)  # units are mm throughout
    return approx.reshape(-1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--output", default="configs/maze_path_auto.csv")
    parser.add_argument("--threshold", type=int, default=110)
    parser.add_argument("--max-line-width-mm", type=float, default=3.0,
                        help="Anything wider is treated as wall")
    parser.add_argument("--min-component-mm", type=float, default=14.0,
                        help="Thin components smaller than this (numbers) are dropped")
    parser.add_argument("--gap-mm", type=float, default=18.0,
                        help="Max gap the trace may jump (line under a wall)")
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)

    state = {"seed": None, "trace": None, "waypoints": None,
             "origin": np.zeros(2), "scale": SCALE_PX_PER_MM}

    def on_mouse(event: int, x: int, y: int, *_rest) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["seed"] = (np.array([x, y], dtype=float) / state["scale"]
                             + state["origin"])
            state["trace"] = None  # retrace on next loop

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cv2.createTrackbar("threshold", WINDOW, args.threshold, 255, lambda _v: None)
    print(__doc__)

    with CameraCapture(config.camera) as camera:
        frame = camera.read().image
        topdown, origin, scale = warp_topdown(frame, homography)
        state["origin"], state["scale"] = origin, scale
        last_threshold = -1
        line_mask = None
        points = None

        def mm_to_px(pt_mm: np.ndarray) -> tuple[int, int]:
            p = (np.asarray(pt_mm) - origin) * scale
            return (int(p[0]), int(p[1]))

        while True:
            threshold = cv2.getTrackbarPos("threshold", WINDOW)
            if threshold != last_threshold:
                line_mask = extract_line_mask(
                    topdown, threshold, args.max_line_width_mm,
                    args.min_component_mm, scale
                )
                points = line_points_mm(line_mask, origin_mm=origin, scale=scale)
                last_threshold = threshold
                state["trace"] = None

            if state["seed"] is not None and state["trace"] is None and len(points):
                trace = trace_line(points, state["seed"], gap_mm=args.gap_mm)
                state["trace"] = trace
                state["waypoints"] = simplify(trace)
                length_mm = float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum())
                print(f"traced {length_mm:.0f} mm of line, "
                      f"simplified to {len(state['waypoints'])} waypoints")

            view = topdown.copy()
            view[line_mask > 0] = (0, 128, 255)  # detected line: orange
            trace = state["trace"]
            if trace is not None and len(trace) >= 2:
                # gradient blue->green shows trace order
                n = len(trace)
                for i in range(n - 1):
                    c = (255 - int(255 * i / n), int(255 * i / n), 0)
                    cv2.line(view, mm_to_px(trace[i]), mm_to_px(trace[i + 1]), c, 2)
                for w in state["waypoints"]:
                    cv2.circle(view, mm_to_px(w), 4, (0, 0, 255), -1)
            msg = ("click the START of the line" if trace is None
                   else f"waypoints: {len(state['waypoints'])}  s=save  r=reset")
            cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.imshow(WINDOW, view)
            cv2.imshow("line mask", line_mask)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                frame = camera.read().image
                topdown, origin, scale = warp_topdown(frame, homography)
                state["origin"], state["scale"] = origin, scale
                last_threshold = -1
            elif key == ord("r"):
                state["seed"] = None
                state["trace"] = None
            elif key == ord("s") and state["waypoints"] is not None:
                out = Path(args.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["x_mm", "y_mm"])
                    for x_mm, y_mm in state["waypoints"]:
                        writer.writerow([f"{x_mm:.1f}", f"{y_mm:.1f}"])
                print(f"saved {len(state['waypoints'])} waypoints -> {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
