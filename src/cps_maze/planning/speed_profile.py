"""Precomputed velocity profile along the route.

Why this exists: the previous design stacked per-frame reactive speed caps
(hole proximity, wall proximity, corner turning) plus a stall kick and an
emergency brake. Where hole capture zones overlap the route, the caps pinned
the crawl speed AT the stall-detection threshold, so the ball's crawl was
mistaken for a stall, the kick launched it, the brake slammed it, forever -
observed as the ball "spazzing" between holes (replay: 15 s to travel 19 mm
at commands up to 0.95).

The maze is STATIC, so the correct speed at every point of the route can be
decided once, coherently, before the run:

1. Sample the route every step_mm and compute a local speed LIMIT at each
   sample from static factors:
   - hole clearance: a smooth ramp down to a committed PASS speed inside
     capture zones. Deliberately not a crawl: slow rolling near a hole
     maximizes exposure time and stiction cycling; a moderate, committed
     speed crosses the pass quickly and stably.
   - wall clearance of the centerline point,
   - accumulated path turning (corners and chicanes),
   - a fixed end speed at the goal.
2. Backward pass: v[i] = min(limit[i], sqrt(v[i+1]^2 + 2 a step)) - every
   slowdown is reachable by braking, so deceleration starts early enough by
   construction.
3. Forward pass: same with the acceleration limit - exits ramp up smoothly.

The runtime controller then simply tracks profile.speed_at(progress): one
smooth, self-consistent plan instead of several reactive caps fighting each
other. The stall detector must sit BELOW the profile minimum (enforced by
the runner) so planned slow rolling is never mistaken for a stall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeedProfile:
    step_mm: float
    speeds_mm_s: np.ndarray

    def speed_at(self, progress_mm: float) -> float:
        idx = progress_mm / self.step_mm
        return float(np.interp(idx, np.arange(len(self.speeds_mm_s)),
                               self.speeds_mm_s))

    def min_speed(self) -> float:
        return float(np.min(self.speeds_mm_s))

    def summary(self) -> str:
        s = self.speeds_mm_s
        return (f"speed profile: {len(s)} samples @ {self.step_mm}mm, "
                f"min {s.min():.0f} / median {np.median(s):.0f} / "
                f"max {s.max():.0f} mm/s")


def build_speed_profile(
    path,
    hole_map=None,
    wall_map=None,
    *,
    v_max_mm_s: float,
    hole_pass_mm_s: float = 16.0,
    hole_slow_band_mm: float = 20.0,
    floor_mm_s: float = 12.0,
    corner_slow_deg: float = 100.0,
    corner_span_mm: float = 15.0,
    corner_noise_deg: float = 12.0,
    corner_min_frac: float = 0.35,
    accel_mm_s2: float = 150.0,
    end_speed_mm_s: float = 10.0,
    step_mm: float = 2.0,
) -> SpeedProfile:
    total = float(path.cumulative_lengths[-1])
    n = max(int(np.ceil(total / step_mm)) + 1, 2)
    limits = np.full(n, float(v_max_mm_s))

    for i in range(n):
        s = min(i * step_mm, total)
        p = path.point_at_progress_mm(s)

        # corners / chicanes
        turn = path.heading_change_deg(s, span_mm=corner_span_mm,
                                       noise_deg=corner_noise_deg)
        corner_scale = max(corner_min_frac,
                           1.0 - turn / max(corner_slow_deg, 1e-9))
        limit = v_max_mm_s * corner_scale

        # wall clearance of the centerline point
        if wall_map is not None:
            limit = min(limit, v_max_mm_s * wall_map.speed_scale(p))

        # hole clearance: committed pass speed inside, smooth ramp outside
        if hole_map is not None:
            clearance = hole_map.clearance_mm(p)
            if clearance <= 0.0:
                limit = min(limit, hole_pass_mm_s)
            elif clearance < hole_slow_band_mm:
                t = clearance / hole_slow_band_mm
                limit = min(limit,
                            hole_pass_mm_s + t * (v_max_mm_s - hole_pass_mm_s))

        limits[i] = max(limit, floor_mm_s)

    limits[-1] = min(limits[-1], end_speed_mm_s)  # arrive gently at the goal

    # backward pass: braking feasibility toward every future constraint
    speeds = limits.copy()
    for i in range(n - 2, -1, -1):
        reachable = np.sqrt(speeds[i + 1] ** 2 + 2.0 * accel_mm_s2 * step_mm)
        speeds[i] = min(speeds[i], reachable)

    # forward pass: acceleration feasibility out of every slow zone
    for i in range(1, n):
        reachable = np.sqrt(speeds[i - 1] ** 2 + 2.0 * accel_mm_s2 * step_mm)
        speeds[i] = min(speeds[i], reachable)

    return SpeedProfile(step_mm=float(step_mm), speeds_mm_s=speeds)
