"""Hole-aware speed limiting and emergency braking.

The holes are known (configs/maze_holes.csv), so the controller can act on
them BEFORE the ball is committed, in two layers:

1. Anticipatory: scan the route ahead for passes near a hole and cap the
   desired speed from braking physics, v_allowed = sqrt(2 * a_brake * d),
   so deceleration starts early enough by construction instead of reacting
   when the hole is already close.

2. Reactive: project the ball's CURRENT velocity forward; if the predicted
   trajectory enters a hole's capture zone and the stopping distance at the
   current speed exceeds the distance to it, the ball cannot stop in time -
   command a full emergency brake opposite to the velocity.
"""
from __future__ import annotations

import numpy as np


class HoleMap:
    def __init__(self, holes: np.ndarray, ball_radius_mm: float = 6.0,
                 margin_mm: float = 4.0):
        """holes: (N, 3) array of x_mm, y_mm, radius_mm."""
        self.holes = np.asarray(holes, dtype=float).reshape(-1, 3)
        # capture radius: the ball falls when its CENTER gets this close
        self.capture_mm = self.holes[:, 2] + ball_radius_mm + margin_mm \
            if len(self.holes) else np.zeros(0)

    def path_hazard_distance_mm(
        self,
        path,
        progress_mm: float,
        horizon_mm: float = 80.0,
        step_mm: float = 4.0,
        ignore_current_hazard: bool = True,
    ) -> float | None:
        """Distance along the route to the next ENTRY into a hole capture zone.

        Returns None if the route ahead is clear within the horizon. When the
        current path point is already inside an inflated capture zone,
        ``ignore_current_hazard`` skips that zone until the route exits it.
        That prevents a planned narrow pass from reporting ``0 mm ahead`` for
        every frame and acting like an invisible braking wall.
        """
        if not len(self.holes):
            return None
        steps = max(int(horizon_mm / step_mm), 1)
        started_in_hazard = bool(self._path_point_in_capture(
            path.point_at_progress_mm(progress_mm)))
        for i in range(steps + 1):
            s = progress_mm + i * step_mm
            p = path.point_at_progress_mm(s)
            in_hazard = self._path_point_in_capture(p)
            if ignore_current_hazard and started_in_hazard:
                if in_hazard:
                    continue
                started_in_hazard = False
                continue
            if in_hazard:
                return float(i * step_mm)
        return None

    def _path_point_in_capture(self, p: np.ndarray) -> bool:
        d = np.hypot(self.holes[:, 0] - p[0], self.holes[:, 1] - p[1])
        return bool(np.any(d < self.capture_mm))

    def clearance_mm(self, p: np.ndarray) -> float:
        """Distance from a point to the nearest capture-zone EDGE.

        Negative when the point is inside a capture zone; +inf with no
        holes. Used to build the route speed profile."""
        if not len(self.holes):
            return float("inf")
        d = np.hypot(self.holes[:, 0] - p[0], self.holes[:, 1] - p[1])
        return float(np.min(d - self.capture_mm))

    def speed_cap_mm_s(
        self,
        hazard_distance_mm: float | None,
        brake_accel_mm_s2: float,
        standoff_mm: float = 10.0,
        floor_mm_s: float = 8.0,
    ) -> float | None:
        """Max safe speed given a hazard ahead: v = sqrt(2 a d), where d is
        the distance remaining before the standoff point. None = no cap."""
        if hazard_distance_mm is None:
            return None
        usable = max(hazard_distance_mm - standoff_mm, 0.0)
        return max(float(np.sqrt(2.0 * brake_accel_mm_s2 * usable)), floor_mm_s)

    def trajectory_hazard(
        self,
        position_mm: np.ndarray,
        velocity_mm_s: np.ndarray,
        horizon_s: float = 0.8,
    ) -> tuple[float, float] | None:
        """If the straight-line projection of the current velocity enters a
        hole's capture zone within the horizon, returns (time_to_entry_s,
        distance_to_entry_mm) for the earliest hole. None otherwise."""
        if not len(self.holes):
            return None
        speed = float(np.linalg.norm(velocity_mm_s))
        if speed < 1e-6:
            return None
        best: tuple[float, float] | None = None
        for (hx, hy, _r), cap in zip(self.holes, self.capture_mm):
            rel = np.array([hx, hy]) - np.asarray(position_mm, dtype=float)
            # closest point of approach of p + v t to the hole center
            t_cpa = float(np.clip(np.dot(rel, velocity_mm_s) / (speed * speed),
                                  0.0, horizon_s))
            closest = rel - velocity_mm_s * t_cpa
            if float(np.linalg.norm(closest)) >= cap:
                continue
            # entry time: solve |rel - v t| = cap (first root before t_cpa)
            a = speed * speed
            b = -2.0 * float(np.dot(rel, velocity_mm_s))
            c = float(np.dot(rel, rel)) - cap * cap
            disc = b * b - 4 * a * c
            if disc <= 0:
                continue
            t_entry = (-b - np.sqrt(disc)) / (2 * a)
            if t_entry < 0.0:
                t_entry = 0.0  # already inside the capture zone
            if t_entry > horizon_s:
                continue
            if best is None or t_entry < best[0]:
                best = (float(t_entry), float(t_entry * speed))
        return best

    def must_emergency_brake(
        self,
        position_mm: np.ndarray,
        velocity_mm_s: np.ndarray,
        brake_accel_mm_s2: float,
        horizon_s: float = 0.8,
        safety_factor: float = 1.3,
    ) -> bool:
        """True when the ball's trajectory enters a hole AND its stopping
        distance (with safety factor) exceeds the distance to entry - i.e.
        normal control can no longer prevent the fall."""
        hazard = self.trajectory_hazard(position_mm, velocity_mm_s, horizon_s)
        if hazard is None:
            return False
        _t_entry, dist_entry = hazard
        speed = float(np.linalg.norm(velocity_mm_s))
        stopping = speed * speed / (2.0 * brake_accel_mm_s2)
        return stopping * safety_factor >= dist_entry


def should_emergency_brake(
    hole_map: HoleMap,
    position_mm: np.ndarray,
    velocity_mm_s: np.ndarray,
    brake_accel_mm_s2: float,
    path_tangent: np.ndarray,
    cross_track_mm: float,
    min_speed_mm_s: float = 15.0,
    offroute_mm: float = 12.0,
    align_deg: float = 40.0,
    horizon_s: float = 0.8,
) -> bool:
    """Emergency-brake only for genuine run-offs, never for planned passes.

    The annotated route legitimately threads between close holes whose
    capture zones overlap the centerline. A ball rolling ALONG the route
    there is a planned pass: the anticipatory speed cap already has it
    crawling, and slamming the emergency brake instead creates a limit
    cycle - brake, stall-kick, brake - that behaves like an invisible wall
    (observed: ball shaking between two holes, or circling). So the
    emergency only fires when the ball is actually LEAVING the route:
    off the centerline, or moving misaligned with the path direction.
    """
    speed = float(np.linalg.norm(velocity_mm_s))
    if speed < min_speed_mm_s:
        return False
    heading = velocity_mm_s / speed
    aligned = float(np.dot(heading, path_tangent)) >= float(
        np.cos(np.radians(align_deg)))
    if aligned and cross_track_mm <= offroute_mm:
        return False  # following the route: the crawl handles the pass
    return hole_map.must_emergency_brake(
        position_mm, velocity_mm_s, brake_accel_mm_s2, horizon_s)
