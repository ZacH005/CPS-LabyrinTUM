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
        self.min_area = float(config["min_blob_area_px"])
        self.max_area = float(config["max_blob_area_px"])
        self.threshold_value = int(config["threshold_value"])
        self.blur_kernel = int(config["blur_kernel"])
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1

    def detect(self, image_bgr: np.ndarray) -> BallDetection:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                candidates.append((area, contour))

        if not candidates:
            return BallDetection(found=False)

        area, contour = max(candidates, key=lambda item: item[0])
        (x, y), radius = cv2.minEnclosingCircle(contour)
        return BallDetection(found=True, x_px=x, y_px=y, radius_px=radius, area_px=area)

    @staticmethod
    def draw_detection(image_bgr: np.ndarray, detection: BallDetection) -> np.ndarray:
        output = image_bgr.copy()
        if detection.found and detection.x_px is not None and detection.y_px is not None:
            center = (int(detection.x_px), int(detection.y_px))
            radius = int(detection.radius_px or 4)
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            cv2.circle(output, center, 2, (0, 0, 255), -1)
        return output

