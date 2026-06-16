#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from cps_maze.calibration.homography import estimate_homography


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--output", default="calibration/board_homography.npz")
    args = parser.parse_args()

    rows = np.genfromtxt(args.points_csv, delimiter=",", names=True)
    image_points = np.column_stack([rows["x_px"], rows["y_px"]])
    board_points = np.column_stack([rows["x_mm"], rows["y_mm"]])
    if image_points.shape[0] < 4:
        raise ValueError("At least four point correspondences are required")

    homography = estimate_homography(image_points, board_points)
    homography.save(args.output)
    print(f"Saved homography to {args.output}")


if __name__ == "__main__":
    main()

