from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp_s: float


class CameraCapture:
    def __init__(self, config: dict):
        self.config = config
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        device_index = int(self.config["device_index"])
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config["width"]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config["height"]))
        self.cap.set(cv2.CAP_PROP_FPS, int(self.config["fps"]))

    def read(self) -> Frame:
        if self.cap is None:
            raise RuntimeError("Camera is not open")
        ok, image = self.cap.read()
        if not ok or image is None:
            raise RuntimeError("Could not read camera frame")
        if self.config.get("flip_horizontal", False):
            image = cv2.flip(image, 1)
        if self.config.get("flip_vertical", False):
            image = cv2.flip(image, 0)
        return Frame(image=image, timestamp_s=monotonic())

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

