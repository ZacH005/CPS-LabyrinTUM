"""Core of the motion + specular-highlight ball tracker.

Extracted from scripts/pipeline.py so the same detection code drives both the
offline video CLI and the live tools (axis_check, run_autonomous). The tracker
is streaming-safe by design: update() only ever looks at the current and
previous frame.

Detection cues (see scripts/pipeline.py docstring for the full rationale):
- motion: frame-to-frame diff blobs of ball-like size/circularity
- specular highlight: the metal ball glints ~254; holes/text top out ~150-215
- static confusers: pre-computed always-bright spots, permanently excluded
- ROI polygon: candidates outside the playable surface are never the ball
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cps_maze.vision.ball_tracker import BallDetection


def motion_candidates(prev_gray, next_gray, min_area=50, max_area=700, min_circularity=0.5):
    """Frame-to-frame diff blobs matching the ball's size/circularity. The
    one cue that works while the ball is actually moving."""
    diff = cv2.GaussianBlur(cv2.absdiff(prev_gray, next_gray), (5, 5), 0)
    _, mask = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < min_circularity:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        out.append((x, y, r))
    return out


def highlight_candidates(gray, thresh=225, min_area=1, max_area=60, ball_r=9):
    """Per-frame near-saturated blobs. Unlike motion_candidates this works
    even when the ball is stationary; callers must gate by proximity to a
    predicted position since it also fires on other shiny objects."""
    mask = (gray >= thresh).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[i]
        out.append((float(cx), float(cy), float(ball_r)))
    return out


def specular_peak(gray, x, y, r):
    """Max brightness in a patch -- the ball/hole discriminator (ball ~254,
    holes/text top out ~150-215)."""
    h, w = gray.shape
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    patch = gray[y0:y1, x0:x1]
    return int(patch.max()) if patch.size else 0


def auto_seed(gray, min_specular=225):
    """Brightest small blob in the frame. Best-effort convenience for a
    static start position -- unreliable when other bright spots (holes,
    glare) outshine the ball, so prefer an explicit seed when you can."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, maxval, _, maxloc = cv2.minMaxLoc(blurred)
    if maxval < min_specular:
        return None
    return float(maxloc[0]), float(maxloc[1])


def in_roi(x, y, roi):
    """roi is a polygon (list of [x,y]) hugging the playable board surface --
    anything outside it (desk clutter, screws, cables) is never a candidate,
    regardless of how bright or how well it moves."""
    if not roi:
        return True
    poly = np.array(roi, dtype=np.float32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def load_calibration(path):
    """Load (confusers, roi) from a JSON written by pipeline.py --calibrate."""
    data = json.loads(Path(path).read_text())
    return [tuple(c) for c in data["confusers"]], data.get("roi")


class BallTracker:
    """Feed grayscale frames one at a time via update(); works for live
    streams too -- it never looks ahead."""

    def __init__(self, seed_xy, seed_r=9, max_jump=35, max_search=60,
                 search_growth=1.15, min_specular=225, max_predict_frames=8,
                 static_confusers=None, roi=None):
        self.pos = np.array(seed_xy, dtype=float)
        self.vel = np.zeros(2)
        self.r = seed_r
        self.max_jump = max_jump
        self.max_search = max_search
        self.search_growth = search_growth
        self.min_specular = min_specular
        self.max_predict_frames = max_predict_frames
        self.miss_streak = 0
        self.prev_gray = None
        # Precomputed once (see compute_static_confusers) and permanently
        # excluded -- this is what tells "always-bright board feature" apart
        # from "the ball, currently sitting still", which a purely reactive
        # stillness check cannot.
        self.static_confusers = static_confusers or []
        self.roi = roi

    def _filter_candidates(self, cands):
        out = []
        for c in cands:
            x, y = c[0], c[1]
            if not in_roi(x, y, self.roi):
                continue
            if any(np.hypot(x - sx, y - sy) <= sr for (sx, sy, sr) in self.static_confusers):
                continue
            out.append(c)
        return out

    def update(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return self.pos[0], self.pos[1], self.r, "seed"

        # union of two independent cues: motion (fast-moving ball) and raw
        # appearance (stationary/slow ball, where motion diff shows nothing)
        cands = motion_candidates(self.prev_gray, gray) + highlight_candidates(
            gray, thresh=self.min_specular, ball_r=self.r
        )
        cands = self._filter_candidates(cands)

        if self.miss_streak <= self.max_predict_frames:
            search_r = min(self.max_jump * (self.search_growth ** self.miss_streak), self.max_search)
            predicted = self.pos + self.vel
            best, best_d = None, None
            for (x, y, r) in cands:
                d = np.hypot(x - predicted[0], y - predicted[1])
                if d > search_r or specular_peak(gray, x, y, r) < self.min_specular:
                    continue
                if best_d is None or d < best_d:
                    best, best_d = (x, y, r), d
        else:
            # long lost: reacquire anywhere via strongest specular hotspot
            best, best_peak = None, -1
            for (x, y, r) in cands:
                peak = specular_peak(gray, x, y, r)
                if peak >= self.min_specular and peak > best_peak:
                    best, best_peak = (x, y, r), peak

        self.prev_gray = gray

        if best is not None:
            new_pos = np.array([best[0], best[1]])
            self.vel = new_pos - self.pos if self.miss_streak == 0 else (new_pos - self.pos) / (self.miss_streak + 1)
            self.pos = new_pos
            self.r = 0.7 * self.r + 0.3 * best[2]
            self.miss_streak = 0
            return self.pos[0], self.pos[1], self.r, "detected"

        self.miss_streak += 1
        if self.miss_streak <= self.max_predict_frames:
            self.pos = self.pos + self.vel  # constant-velocity coast through the gap
            return self.pos[0], self.pos[1], self.r, "predicted"

        return self.pos[0], self.pos[1], self.r, "lost"


class PipelineBallTracker:
    """Live adapter: BGR frames in, BallDetection out.

    Wraps BallTracker behind the same interface as BrightBlobBallTracker so
    axis_check / run_autonomous / check_camera can switch tracker via config.
    Self-seeds from the first frame with a strong specular blob and re-seeds
    the same way if the track is lost for a long stretch.

    Optional vision-config keys (all have working defaults):
      min_specular       specular gate, default 225 (lower under dim lighting)
      seed_r             initial ball radius px, default 9
      confusers_file     JSON from pipeline.py --calibrate; loaded if present
    """

    def __init__(self, config: dict):
        self.min_specular = int(config.get("min_specular", 225))
        self.seed_r = float(config.get("seed_r", 9))
        self.confusers: list = []
        self.roi = None
        confusers_file = config.get("confusers_file", "calibration/live_confusers.json")
        if confusers_file and Path(confusers_file).exists():
            self.confusers, self.roi = load_calibration(confusers_file)
        self.tracker: BallTracker | None = None

    def _make_tracker(self, seed_xy) -> BallTracker:
        return BallTracker(
            seed_xy,
            seed_r=self.seed_r,
            min_specular=self.min_specular,
            static_confusers=self.confusers,
            roi=self.roi,
        )

    def seed(self, x_px: float, y_px: float) -> None:
        """Manually (re)seed the track, e.g. from a user click on the ball.

        Far more reliable than auto-seed on a bright board where glare spots
        can outshine the ball."""
        self.tracker = self._make_tracker((float(x_px), float(y_px)))

    def detect(self, image_bgr: np.ndarray) -> BallDetection:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        if self.tracker is None:
            seed = auto_seed(gray, self.min_specular)
            if seed is None:
                return BallDetection(found=False)
            self.tracker = self._make_tracker(seed)

        x, y, r, status = self.tracker.update(gray)
        if status == "lost":
            # allow a fresh auto-seed once the ball is visible again
            seed = auto_seed(gray, self.min_specular)
            if seed is not None and in_roi(seed[0], seed[1], self.roi):
                self.tracker = self._make_tracker(seed)
            return BallDetection(found=False)
        return BallDetection(found=True, x_px=float(x), y_px=float(y),
                             radius_px=float(r), area_px=float(np.pi * r * r))

    @staticmethod
    def draw_detection(image_bgr: np.ndarray, detection: BallDetection) -> np.ndarray:
        output = image_bgr.copy()
        if detection.found and detection.x_px is not None and detection.y_px is not None:
            center = (int(detection.x_px), int(detection.y_px))
            radius = int(detection.radius_px or 4)
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            cv2.circle(output, center, 2, (0, 0, 255), -1)
        return output


def make_tracker(vision_config: dict):
    """Factory: vision.tracker selects the live detector.

    "pipeline" (default) = motion+specular BallTracker; "bright_blob" = the
    original threshold detector.
    """
    from cps_maze.vision.ball_tracker import BrightBlobBallTracker

    name = str(vision_config.get("tracker", "pipeline")).lower()
    if name == "bright_blob":
        return BrightBlobBallTracker(vision_config)
    if name == "pipeline":
        return PipelineBallTracker(vision_config)
    raise ValueError(f"unknown vision.tracker: {name!r} (use 'pipeline' or 'bright_blob')")
