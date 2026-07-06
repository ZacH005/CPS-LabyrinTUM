from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BallDetection:
    found: bool
    x_px: float | None = None
    y_px: float | None = None
    radius_px: float | None = None
    area_px: float | None = None


class BrightBlobBallTracker:
    """Initial tracker for a reflective silver marble under controlled lighting."""

    def __init__(self, config: dict):
        self.min_area = 25.0
        self.max_area = 5000.0
        self.threshold_value = 220
        self.smoothing_alpha = 0.6
        self.last_pos = None
        self.last_radius = None
        self.blur_kernel = 5
        self.use_otsu = False
        self.clahe_clip = 2.0
        self.morph_kernel = 5
        #debug overlay
        self.debug_enabled = False
        self.debug_every_n = 1
        self.debug_start_stage = "binary"
        self.frame_count = 0
        self.debug_frames: dict[str, np.ndarray] = {}
        self.debug_candidates: list[dict] = []
        self.debug_chosen: dict | None = None
        self.update_from_config(config)

    def update_from_config(self, config: dict) -> None:
        self.min_area = float(config.get("min_blob_area_px", self.min_area))
        self.max_area = float(config.get("max_blob_area_px", self.max_area))
        self.threshold_value = int(config.get("threshold_value", self.threshold_value))
        self.smoothing_alpha = float(config.get("smoothing_alpha", self.smoothing_alpha))
        self.blur_kernel = int(config.get("blur_kernel", self.blur_kernel))
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.use_otsu = bool(config.get("use_otsu", self.use_otsu))
        self.clahe_clip = float(config.get("clahe_clip", self.clahe_clip))
        self.morph_kernel = int(config.get("morph_kernel", self.morph_kernel))
        self.debug_enabled = bool(config.get("debug_overlay", self.debug_enabled))
        self.debug_every_n = int(config.get("debug_overlay_every_n", self.debug_every_n))
        self.debug_start_stage = str(config.get("debug_start_stage", self.debug_start_stage))

    def detect(self, image_bgr: np.ndarray) -> BallDetection:
        self.frame_count += 1
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        norm = clahe.apply(gray)
        blurred = cv2.GaussianBlur(norm, (self.blur_kernel, self.blur_kernel), 0)

        # fixed threshold (preferred) and also compute adaptive variants for debug
        if self.use_otsu:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)

        adaptive_mean = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
        adaptive_gauss = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

        # morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

        # save debug frames
        self.debug_frames = {
            "gray": gray,
            "clahe": norm,
            "blurred": blurred,
            "binary": binary,
            "adaptive_mean": adaptive_mean,
            "adaptive_gauss": adaptive_gauss,
            "morph": morph,
        }

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[float, float, float, float]] = []  # (score, cx, cy, radius)
        eps = 1e-6
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + eps)

            if circularity < 0.6:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)

            # mean intensity inside contour (use blurred/normalized image)
            mask = np.zeros(blurred.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_intensity = float(cv2.mean(blurred, mask=mask)[0]) / 255.0

            # distance penalty to previous detection (if available)
            dist_penalty = 0.0
            if getattr(self, "last_pos", None) is not None:
                dx = cx - self.last_pos[0]
                dy = cy - self.last_pos[1]
                dist = np.hypot(dx, dy)
                # normalize by expected radius to get a unitless penalty
                denom = max(1.0, getattr(self, "last_radius", radius))
                dist_penalty = dist / denom

            # simple scoring: higher circularity, larger area, brighter, nearer previous pos
            score = (2.0 * circularity) + (area / (self.max_area + eps)) + (0.8 * mean_intensity) - (0.6 * dist_penalty)
            
            candidates.append((score, float(cx), float(cy), float(radius), area, circularity))

        if not candidates:
            return BallDetection(found=False)
        
        # pick best-scoring candidate
        best = max(candidates, key=lambda t: t[0])
        _, bx, by, br, barea, bcirc = best
        
        # update temporal state (exponential smoothing)
        alpha = getattr(self, "smoothing_alpha", 0.6)
        if getattr(self, "last_pos", None) is None:
            self.last_pos = (bx, by)
            self.last_radius = br
        else:
            lx, ly = self.last_pos
            self.last_pos = (alpha * bx + (1 - alpha) * lx, alpha * by + (1 - alpha) * ly)
            self.last_radius = alpha * br + (1 - alpha) * getattr(self, "last_radius", br)

        # return smoothed detection
        sx, sy = self.last_pos
        sr = self.last_radius
        return BallDetection(found=True, x_px=sx, y_px=sy, radius_px=sr, area_px=barea)

        # area, contour = max(candidates, key=lambda item: item[0])
        # (x, y), radius = cv2.minEnclosingCircle(contour)
        # return BallDetection(found=True, x_px=x, y_px=y, radius_px=radius, area_px=area)

    @staticmethod
    def draw_detection(image_bgr: np.ndarray, detection: BallDetection) -> np.ndarray:
        output = image_bgr.copy()
        if detection.found and detection.x_px is not None and detection.y_px is not None:
            center = (int(detection.x_px), int(detection.y_px))
            radius = int(detection.radius_px or 4)
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            cv2.circle(output, center, 2, (0, 0, 255), -1)
        return output

    def draw_debug_overlay(self, image_bgr: np.ndarray, stage: str | None = None) -> np.ndarray:
        """Draw debug overlay on a copy of the input image.

        stage: which preprocessing stage to show in the inset. One of keys of self.debug_frames.
        """
        overlay = image_bgr.copy()
        h, w = overlay.shape[:2]

        # choose stage
        stage = stage or self.debug_start_stage
        stage_img = self.debug_frames.get(stage)
        if stage_img is None:
            stage_img = self.debug_frames.get("morph")

        # prepare inset
        if stage_img is not None:
            inset = cv2.resize(stage_img, (w // 4, h // 4))
            if len(inset.shape) == 2:
                inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
            ih, iw = inset.shape[:2]
            overlay[5 : 5 + ih, w - 5 - iw : w - 5] = inset
            cv2.rectangle(overlay, (w - 5 - iw, 5), (w - 5, 5 + ih), (0, 255, 255), 1)
            cv2.putText(overlay, stage, (w - 5 - iw, 5 + ih + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # draw candidates
        for cand in self.debug_candidates:
            cx = int(cand["cx"])
            cy = int(cand["cy"])
            r = int(cand["r"])
            score = cand["score"]
            cv2.circle(overlay, (cx, cy), max(2, int(r)), (255, 0, 0), 1)
            cv2.putText(overlay, f"{score:.2f}", (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # highlight chosen
        if self.debug_chosen is not None:
            cx = int(self.debug_chosen["cx"])
            cy = int(self.debug_chosen["cy"])
            r = int(self.debug_chosen["r"])
            cv2.circle(overlay, (cx, cy), max(2, r), (0, 255, 0), 2)
            cv2.circle(overlay, (cx, cy), 2, (0, 0, 255), -1)
            cv2.putText(overlay, f"score {self.debug_chosen['score']:.2f}", (cx + 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # diagnostics box
        info_lines = [f"Frame: {self.frame_count}", f"Candidates: {len(self.debug_candidates)}"]
        y0 = h - 10 - (len(info_lines) * 14)
        for i, line in enumerate(info_lines):
            y = y0 + i * 14
            cv2.rectangle(overlay, (5, y - 12), (180, y + 4), (0, 0, 0), -1)
            cv2.putText(overlay, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return overlay

