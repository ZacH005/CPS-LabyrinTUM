#!/usr/bin/env python3
"""Diagnose an autonomous run from its CSV log.

Reports where the run spent its time, where the ball stalled while being
commanded (stiction events), and how far it strayed from the path -- so
tuning is driven by measurements instead of impressions.

    python scripts/analyze_run.py data/raw/autonomous_run.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cps_maze.config import load_config
from cps_maze.planning.path import WaypointPath


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [row for row in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="data/raw/autonomous_run.csv")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stall-speed", type=float, default=8.0,
                        help="mm/s below which the ball counts as stalled")
    parser.add_argument("--stall-cmd", type=float, default=0.05,
                        help="min |command| for a stall to count as 'commanded but stuck'")
    args = parser.parse_args()

    config = load_config(args.config)
    path = WaypointPath.from_csv(config.resolve_path(config.maze["path_file"]))
    total_mm = float(path.cumulative_lengths[-1])

    rows = load_rows(Path(args.log))
    if not rows:
        raise SystemExit("empty log")

    found = [r for r in rows if r["found"] in ("True", "true", "1")]
    t = np.array([float(r["timestamp_s"]) for r in found])
    pos = np.array([[float(r["x_mm"]), float(r["y_mm"])] for r in found])
    vel = np.array([[float(r["vx_mm_s"]), float(r["vy_mm_s"])] for r in found])
    cmd = np.array([[float(r["yaw_command"]), float(r["pitch_command"])] for r in found])
    progress = np.array([
        float(r["progress_mm"]) if r.get("progress_mm") not in (None, "",)
        else path.nearest_progress_mm(p)
        for r, p in zip(found, pos)
    ])

    duration = t[-1] - t[0] if len(t) > 1 else 0.0
    speed = np.linalg.norm(vel, axis=1)
    cmd_mag = np.linalg.norm(cmd, axis=1)

    # cross-track error: distance from ball to its projection on the path
    cross = np.array([
        float(np.linalg.norm(p - path.point_at_progress_mm(g)))
        for p, g in zip(pos, progress)
    ])

    # stall episodes: commanded but not moving
    stalled = (speed < args.stall_speed) & (cmd_mag > args.stall_cmd)
    episodes = []
    start = None
    for i, s in enumerate(stalled):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if t[i] - t[start] >= 0.4:
                episodes.append((t[start] - t[0], t[i] - t[start],
                                 progress[start], cross[start]))
            start = None
    if start is not None and t[-1] - t[start] >= 0.4:
        episodes.append((t[start] - t[0], t[-1] - t[start],
                         progress[start], cross[start]))

    print(f"log: {args.log}")
    print(f"frames: {len(rows)} total, {len(found)} with ball "
          f"({100 * len(found) / len(rows):.0f}% detection rate)")
    print(f"duration: {duration:.1f}s")
    print(f"path progress: {progress.min():.0f} -> {progress.max():.0f} mm "
          f"of {total_mm:.0f} mm ({100 * progress.max() / total_mm:.0f}% reached)")
    print(f"speed: median {np.median(speed):.0f} mm/s, p90 {np.percentile(speed, 90):.0f} mm/s")
    print(f"cross-track error: median {np.median(cross):.1f} mm, "
          f"p90 {np.percentile(cross, 90):.1f} mm, max {cross.max():.1f} mm")
    print(f"time stalled while commanded: "
          f"{100 * np.mean(stalled):.0f}% of tracked frames")

    if episodes:
        print(f"\nstall episodes (>= 0.4s), longest first:")
        for t0, dur, prog, cerr in sorted(episodes, key=lambda e: -e[1])[:10]:
            print(f"  at t={t0:6.1f}s  for {dur:4.1f}s  at path {prog:5.0f} mm "
                  f"({100 * prog / total_mm:3.0f}%)  cross-track {cerr:.1f} mm")
        print("\nThe 'at path X mm' column tells you WHERE on the maze it sticks -"
              "\nrecurring spots are corner/wall sections worth re-annotating or"
              "\ntuning (denser waypoints, smaller lookahead), not gain problems.")
    else:
        print("\nno stall episodes - if the run still failed, look at cross-track"
              "\nerror (gains) or detection rate (tracking).")


if __name__ == "__main__":
    main()
