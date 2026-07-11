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


def load_roi_file(path):
    """Load an ROI polygon from a JSON written by select_maze_roi.py."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    if "roi" not in data:
        raise ValueError(f"{path} does not contain an roi field")
    return data["roi"]


class BallTracker:
    """Feed grayscale frames one at a time via update(); works for live
    streams too -- it never looks ahead."""

    def __init__(self, seed_xy, seed_r=9, max_jump=35, max_search=60,
                 search_growth=1.15, min_specular=225, max_predict_frames=8,
                 static_confusers=None, roi=None, max_velocity_px_per_frame=35,
                 max_single_frame_jump_px=45, allow_global_reacquire=True,
                 motion_min_specular=None):
        self.pos = np.array(seed_xy, dtype=float)
        self.vel = np.zeros(2)
        self.r = seed_r
        self.max_jump = max_jump
        self.max_search = max_search
        self.search_growth = search_growth
        self.min_specular = min_specular
        # Motion blur smears the glint exactly when the ball moves fast, so
        # motion-cue candidates (already strong evidence: a moving, round,
        # ball-sized blob) get a lower brightness gate than raw highlights.
        self.motion_min_specular = (motion_min_specular if motion_min_specular
                                    is not None else max(min_specular - 50, 150))
        self.max_predict_frames = max_predict_frames
        self.max_velocity_px_per_frame = max_velocity_px_per_frame
        self.max_single_frame_jump_px = max_single_frame_jump_px
        self.allow_global_reacquire = allow_global_reacquire
        self.miss_streak = 0
        self.prev_gray = None
        self.debug: dict = {"status": "seed"}  # per-frame diagnostics
        # Precomputed once (see compute_static_confusers) and permanently
        # excluded -- this is what tells "always-bright board feature" apart
        # from "the ball, currently sitting still", which a purely reactive
        # stillness check cannot.
        self.static_confusers = []
        for sx, sy, sr in static_confusers or []:
            if np.hypot(self.pos[0] - sx, self.pos[1] - sy) <= sr + 2 * seed_r:
                continue
            self.static_confusers.append((sx, sy, sr))
        self.roi = roi

    def _bounded_velocity(self):
        speed = float(np.linalg.norm(self.vel))
        if self.max_velocity_px_per_frame <= 0 or speed <= self.max_velocity_px_per_frame:
            return self.vel
        return self.vel * (self.max_velocity_px_per_frame / speed)

    def _accepts_motion(self, x, y, predicted, search_r):
        """Reject candidates that require an implausible jump.

        The predicted-position gate catches ordinary association errors. The
        last-position gate is a hard safety backstop for cases where a previous
        bad velocity would otherwise move the prediction toward a false glint.
        """
        from_prediction = float(np.hypot(x - predicted[0], y - predicted[1]))
        if from_prediction > search_r:
            return False
        if self.max_single_frame_jump_px > 0:
            allowed_from_last = self.max_single_frame_jump_px * max(1, self.miss_streak + 1)
            from_last = float(np.hypot(x - self.pos[0], y - self.pos[1]))
            if from_last > allowed_from_last:
                return False
        return True

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
            self.debug = {"status": "seed"}
            return self.pos[0], self.pos[1], self.r, "seed"

        # union of two independent cues: motion (fast-moving ball) and raw
        # appearance (stationary/slow ball, where motion diff shows nothing);
        # candidates carry their source so each cue can use its own gate
        motion = motion_candidates(self.prev_gray, gray)
        highlight = highlight_candidates(gray, thresh=self.min_specular, ball_r=self.r)
        raw_cands = ([(x, y, r, "motion") for (x, y, r) in motion]
                     + [(x, y, r, "highlight") for (x, y, r) in highlight])
        cands = self._filter_candidates(raw_cands)

        # Per-frame diagnostics: which stage rejected the ball this frame?
        # Consumed by scripts/debug_tracking.py; costs nothing measurable.
        self.debug = {
            "n_motion": len(motion),
            "n_highlight": len(highlight),
            "n_rej_roi_confuser": len(raw_cands) - len(cands),
            "n_rej_jump": 0,
            "n_rej_specular": 0,
            "search_r": 0.0,
            "peak_at_track": specular_peak(gray, self.pos[0], self.pos[1],
                                           max(self.r, 6)),
            "miss_streak": self.miss_streak,
            "candidates": cands,
        }

        if self.miss_streak <= self.max_predict_frames:
            search_r = min(
                self.max_jump * (self.search_growth ** self.miss_streak),
                self.max_search,
            )
            self.debug["search_r"] = float(search_r)
            predicted = self.pos + self._bounded_velocity()
            best, best_d = None, None
            for (x, y, r, source) in cands:
                d = np.hypot(x - predicted[0], y - predicted[1])
                if not self._accepts_motion(x, y, predicted, search_r):
                    self.debug["n_rej_jump"] += 1
                    continue
                required = (self.motion_min_specular if source == "motion"
                            else self.min_specular)
                if specular_peak(gray, x, y, r) < required:
                    self.debug["n_rej_specular"] += 1
                    continue
                if best_d is None or d < best_d:
                    best, best_d = (x, y, r), d
        elif self.allow_global_reacquire:
            # long lost: reacquire anywhere via strongest specular hotspot
            best, best_peak = None, -1
            for (x, y, r, _source) in cands:
                peak = specular_peak(gray, x, y, r)
                if peak >= self.min_specular and peak > best_peak:
                    best, best_peak = (x, y, r), peak
        else:
            best = None

        self.prev_gray = gray

        if best is not None:
            new_pos = np.array([best[0], best[1]])
            if self.miss_streak == 0:
                self.vel = new_pos - self.pos
            else:
                self.vel = (new_pos - self.pos) / (self.miss_streak + 1)
            self.vel = self._bounded_velocity()
            self.pos = new_pos
            self.r = 0.7 * self.r + 0.3 * best[2]
            self.miss_streak = 0
            self.debug["status"] = "detected"
            return self.pos[0], self.pos[1], self.r, "detected"

        self.miss_streak += 1
        self.debug["miss_streak"] = self.miss_streak
        if self.miss_streak <= self.max_predict_frames:
            self.pos = self.pos + self._bounded_velocity()
            self.debug["status"] = "predicted"
            return self.pos[0], self.pos[1], self.r, "predicted"

        self.debug["status"] = "lost"
        return self.pos[0], self.pos[1], self.r, "lost"


class PipelineBallTracker:
    """Live adapter: BGR frames in, BallDetection out.

    Wraps BallTracker behind the same interface as BrightBlobBallTracker so
    axis_check / run_autonomous / check_camera can switch tracker via config.
    Self-seeds from the first frame with a strong specular blob and re-seeds
    the same way if the track is lost for a long stretch.

    When seeded from a click, it also captures a small appearance template of
    the ball and correlates it near the predicted position every frame. On a
    bright board where glare rivals the ball's glint, this appearance cue is
    far more specific than brightness and rescues the track when the
    motion/specular cues fail (e.g. a slow ball with a weak glint).

    Optional vision-config keys (all have working defaults):
      min_specular             specular gate, default 225 (lower under dim lighting)
      seed_r                   initial ball radius px, default 9
      confusers_file           JSON from pipeline.py --calibrate; loaded if present
      auto_seed                allow initial auto-seed before a click, default false
      auto_reseed              allow auto-reseed after loss, default false
      report_predicted         return found=True for predicted frames, default false
      max_jump_px              predicted-position candidate gate, default 35
      max_search_px            upper bound for expanded search gate, default 60
      max_predict_frames       internal coasting frames after misses, default 2
      max_velocity_px_per_frame velocity clamp for prediction, default 35
      max_single_frame_jump_px hard gate from the last accepted position, default 45
      template_min_score       matchTemplate acceptance (0-1), default 0.55
      template_search_px       template search half-window, default 45
      template_max_correction_px max template correction from track, default 30
      roi_file                 optional ROI JSON; overrides ROI embedded in confusers_file
    """

    def __init__(self, config: dict):
        self.min_specular = int(config.get("min_specular", 225))
        self.motion_min_specular = int(config.get(
            "motion_min_specular", max(self.min_specular - 50, 150)))
        self.seed_r = float(config.get("seed_r", 9))
        self.auto_seed = bool(config.get("auto_seed", False))
        self.auto_reseed = bool(config.get("auto_reseed", False))
        self.report_predicted = bool(config.get("report_predicted", False))
        self.max_jump_px = float(config.get("max_jump_px", 35))
        self.max_search_px = float(config.get("max_search_px", 60))
        self.search_growth = float(config.get("search_growth", 1.15))
        self.max_predict_frames = int(config.get("max_predict_frames", 2))
        self.max_velocity_px_per_frame = float(config.get("max_velocity_px_per_frame", 35))
        self.max_single_frame_jump_px = float(config.get("max_single_frame_jump_px", 45))
        self.allow_global_reacquire = bool(config.get("allow_global_reacquire", False))
        self.template_min_score = float(config.get("template_min_score", 0.55))
        self.template_search_px = int(config.get("template_search_px", 45))
        self.template_max_correction_px = float(config.get("template_max_correction_px", 30))
        self.confusers: list = []
        self.roi = None
        confusers_file = config.get("confusers_file", "calibration/live_confusers.json")
        if confusers_file and Path(confusers_file).exists():
            self.confusers, self.roi = load_calibration(confusers_file)
            if len(self.confusers) > 30:
                print(f"WARNING: {len(self.confusers)} static confusers loaded "
                      f"from {confusers_file} - a healthy board has ~5-15. "
                      "This usually means the BALL was in the calibration "
                      "video and blacklisted its own resting spots; candidates "
                      "inside those zones are permanently rejected (= random "
                      "track losses). Re-record the calibration video with NO "
                      "ball on the board and rerun pipeline.py --calibrate.")
        roi_file = config.get("roi_file")
        if roi_file and Path(roi_file).exists():
            self.roi = load_roi_file(roi_file)
        self.tracker: BallTracker | None = None
        self._last_gray: np.ndarray | None = None
        self._template: np.ndarray | None = None
        self.last_template_score: float = float("nan")  # diagnostics
        self.template_rescued: bool = False  # diagnostics

    def _make_tracker(self, seed_xy) -> BallTracker:
        return BallTracker(
            seed_xy,
            seed_r=self.seed_r,
            max_jump=self.max_jump_px,
            max_search=self.max_search_px,
            search_growth=self.search_growth,
            min_specular=self.min_specular,
            motion_min_specular=self.motion_min_specular,
            max_predict_frames=self.max_predict_frames,
            static_confusers=self.confusers,
            roi=self.roi,
            max_velocity_px_per_frame=self.max_velocity_px_per_frame,
            max_single_frame_jump_px=self.max_single_frame_jump_px,
            allow_global_reacquire=self.allow_global_reacquire,
        )

    def seed(self, x_px: float, y_px: float) -> None:
        """Manually (re)seed the track, e.g. from a user click on the ball.

        Far more reliable than auto-seed on a bright board where glare spots
        can outshine the ball. Also captures an appearance template of the
        ball from the last seen frame for per-frame correlation."""
        self.tracker = self._make_tracker((float(x_px), float(y_px)))
        self._template = None
        if self._last_gray is not None:
            half = max(int(self.seed_r * 1.6), 10)
            h, w = self._last_gray.shape
            x0, x1 = int(x_px) - half, int(x_px) + half + 1
            y0, y1 = int(y_px) - half, int(y_px) + half + 1
            if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
                patch = self._last_gray[y0:y1, x0:x1]
                if int(patch.max()) >= self.min_specular:
                    self._template = patch.copy()

    def _match_template(self, gray: np.ndarray, near_xy, window: int | None = None):
        """Correlate the seed template near a position. Returns (x, y) of the
        best match center, or None if it is not convincing."""
        if self._template is None:
            return None
        if window is None:
            window = self.template_search_px
        th, tw = self._template.shape
        h, w = gray.shape
        cx, cy = int(near_xy[0]), int(near_xy[1])
        x0, x1 = max(0, cx - window), min(w, cx + window)
        y0, y1 = max(0, cy - window), min(h, cy + window)
        if x1 - x0 <= tw or y1 - y0 <= th:
            return None
        result = cv2.matchTemplate(gray[y0:y1, x0:x1], self._template,
                                   cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        self.last_template_score = float(max_val)  # diagnostics
        if max_val < self.template_min_score:
            return None
        return (float(x0 + max_loc[0] + tw / 2), float(y0 + max_loc[1] + th / 2))

    def detect(self, image_bgr: np.ndarray) -> BallDetection:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self._last_gray = gray

        if self.tracker is None:
            if not self.auto_seed:
                return BallDetection(found=False)
            seed = auto_seed(gray, self.min_specular)
            if seed is None or not in_roi(seed[0], seed[1], self.roi):
                return BallDetection(found=False)
            self.tracker = self._make_tracker(seed)

        x, y, r, status = self.tracker.update(gray)

        # appearance assist: correlate the seed template near the track. It
        # rescues "lost"/"predicted" states and corrects false locks (a glint
        # matches brightness but not the ball's appearance).
        self.last_template_score = float("nan")
        self.template_rescued = False
        hit = self._match_template(gray, (x, y))
        if hit is not None:
            correction = float(np.hypot(hit[0] - x, hit[1] - y))
            drifted = correction > 3 * r
            if correction <= self.template_max_correction_px and (
                    status in ("lost", "predicted") or drifted):
                self.tracker.pos = np.array(hit, dtype=float)
                self.tracker.miss_streak = 0
                x, y, status = hit[0], hit[1], "detected"
                self.template_rescued = True

        if status == "lost":
            if self.auto_reseed:
                seed = auto_seed(gray, self.min_specular)
            else:
                seed = None
            if seed is not None and in_roi(seed[0], seed[1], self.roi):
                self.tracker = self._make_tracker(seed)
            return BallDetection(found=False)
        if status == "predicted" and not self.report_predicted:
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
