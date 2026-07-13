from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np

from cps_maze.vision.ball_pipeline import (
    PipelineBallTracker,
    highlight_candidates,
    in_roi,
    motion_candidates,
)

SUPPORTED_VIEWS = {
    "raw",
    "overlay",
    "motion",
    "specular",
    "boundaries",
    "candidates",
}


def default_recording_dir(base_dir: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / f"autonomous_run_{timestamp}"


def default_codec_for_suffix(suffix: str) -> str:
    if suffix.lower() in {".mp4", ".m4v"}:
        return "mp4v"
    return "MJPG"


def parse_record_views(value: str | list[str] | tuple[str, ...] | None) -> set[str]:
    if value is None:
        return {"raw", "overlay"}
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
    else:
        parts = [str(part).strip().lower() for part in value]
    views = {part for part in parts if part}
    unknown = views - SUPPORTED_VIEWS
    if unknown:
        raise ValueError(
            f"unknown recording view(s): {', '.join(sorted(unknown))}; "
            f"use any of {', '.join(sorted(SUPPORTED_VIEWS))}"
        )
    return views


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def draw_text(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    cv2.putText(
        out,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def draw_candidates(
    image: np.ndarray,
    candidates: list[tuple],
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    out = image.copy()
    for cand in candidates:
        x, y, r = float(cand[0]), float(cand[1]), float(cand[2])
        center = (int(round(x)), int(round(y)))
        cv2.circle(out, center, max(int(round(r)), 3), color, 2)
        cv2.putText(
            out,
            label,
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def tracker_roi(tracker: object) -> list | None:
    return getattr(tracker, "roi", None) if isinstance(tracker, PipelineBallTracker) else None


def tracker_confusers(tracker: object) -> list:
    if not isinstance(tracker, PipelineBallTracker):
        return []
    return list(getattr(tracker, "confusers", []))


def draw_boundaries(image: np.ndarray, tracker: object) -> np.ndarray:
    out = image.copy()
    roi = tracker_roi(tracker)
    if roi:
        pts = np.array(roi, dtype=np.int32)
        cv2.polylines(out, [pts], True, (255, 255, 0), 2)
    for sx, sy, sr in tracker_confusers(tracker):
        cv2.circle(out, (int(round(sx)), int(round(sy))), int(round(sr)), (0, 0, 255), 2)
    return draw_text(out, "cyan = ROI boundary, red = static confusers", (255, 255, 0))


def filtered_candidates(
    raw_candidates: list[tuple],
    tracker: object,
) -> tuple[list[tuple], list[tuple]]:
    roi = tracker_roi(tracker)
    confusers = tracker_confusers(tracker)
    kept = []
    rejected = []
    for cand in raw_candidates:
        x, y = cand[0], cand[1]
        rejected_by_roi = not in_roi(x, y, roi)
        rejected_by_confuser = any(
            np.hypot(x - sx, y - sy) <= sr for (sx, sy, sr) in confusers)
        if rejected_by_roi or rejected_by_confuser:
            rejected.append(cand)
        else:
            kept.append(cand)
    return kept, rejected


class VideoStreamWriter:
    def __init__(self, path: Path, fps: float, codec: str):
        self.path = path
        self.fps = float(fps)
        self.codec = codec
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None

    def write(self, image: np.ndarray) -> None:
        image = to_bgr(image)
        height, width = image.shape[:2]
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {self.path}")
            self.size = (width, height)
        elif self.size != (width, height):
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(image)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


@dataclass(frozen=True)
class RunRecordingConfig:
    output_dir: Path
    views: set[str]
    fps: float
    codec: str
    suffix: str = ".avi"
    metadata: dict[str, Any] | None = None


class RunVideoRecorder:
    """Write synchronized run videos plus per-frame timestamps.

    The videos are intentionally frame-index aligned: frame N in every enabled
    view corresponds to row N in frames.csv. Timestamp truth lives in frames.csv,
    because OpenCV VideoWriter stores constant-FPS video even when the control
    loop has jitter.
    """

    def __init__(self, config: RunRecordingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.views = set(config.views)
        self.writers = {
            view: VideoStreamWriter(
                self.output_dir / f"{view}{config.suffix}",
                config.fps,
                config.codec,
            )
            for view in sorted(self.views)
        }
        self.prev_gray: np.ndarray | None = None
        self.frame_index = 0
        self.started_at = datetime.now(timezone.utc)
        self.start_monotonic = monotonic()
        self.outcome = ""
        self.frames_file = None
        self.frames_writer: csv.DictWriter | None = None

    def wants(self, view: str) -> bool:
        return view in self.views

    def set_outcome(self, outcome: str) -> None:
        self.outcome = outcome

    def __enter__(self) -> "RunVideoRecorder":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_file = (self.output_dir / "frames.csv").open(
            "w", newline="", encoding="utf-8")
        self.frames_writer = csv.DictWriter(
            self.frames_file,
            fieldnames=[
                "frame_index",
                "timestamp_s",
                "elapsed_s",
                "armed",
                "found",
                "status",
            ],
        )
        self.frames_writer.writeheader()
        return self

    def _diagnostic_images(self, image: np.ndarray, tracker: object) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            motion_mask = np.zeros_like(gray)
            motion = []
        else:
            diff = cv2.GaussianBlur(cv2.absdiff(self.prev_gray, gray), (5, 5), 0)
            _, motion_mask = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            motion = motion_candidates(self.prev_gray, gray)

        min_specular = int(getattr(tracker, "min_specular", 225))
        specular_mask = (gray >= min_specular).astype(np.uint8) * 255
        highlights = highlight_candidates(gray, thresh=min_specular)
        raw = ([(x, y, r, "motion") for x, y, r in motion]
               + [(x, y, r, "highlight") for x, y, r in highlights])
        kept, rejected = filtered_candidates(raw, tracker)

        candidates = draw_candidates(image, rejected, (0, 0, 255), "X")
        candidates = draw_candidates(candidates, kept, (0, 255, 255), "K")
        candidates = draw_boundaries(candidates, tracker)
        candidates = draw_text(
            candidates,
            f"candidate gate: kept={len(kept)} rejected={len(rejected)}",
            (0, 255, 255),
        )

        return {
            "motion": draw_text(to_bgr(motion_mask), "motion mask", (0, 165, 255)),
            "specular": draw_text(
                to_bgr(specular_mask),
                f"specular mask: gray >= {min_specular}",
                (255, 255, 0),
            ),
            "boundaries": draw_boundaries(image, tracker),
            "candidates": candidates,
        }

    def write(
        self,
        image: np.ndarray,
        timestamp_s: float,
        *,
        tracker: object,
        overlay: np.ndarray | None = None,
        armed: bool = True,
        found: bool = False,
        status: str = "",
    ) -> None:
        diagnostics: dict[str, np.ndarray] | None = None
        if "raw" in self.views:
            self.writers["raw"].write(image)
        if "overlay" in self.views:
            self.writers["overlay"].write(overlay if overlay is not None else image)

        diagnostic_views = self.views & {"motion", "specular", "boundaries", "candidates"}
        if diagnostic_views:
            diagnostics = self._diagnostic_images(image, tracker)
            for view in sorted(diagnostic_views):
                self.writers[view].write(diagnostics[view])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        assert self.frames_writer is not None
        self.frames_writer.writerow({
            "frame_index": self.frame_index,
            "timestamp_s": f"{timestamp_s:.9f}",
            "elapsed_s": f"{monotonic() - self.start_monotonic:.9f}",
            "armed": bool(armed),
            "found": bool(found),
            "status": status,
        })
        assert self.frames_file is not None
        self.frames_file.flush()
        self.frame_index += 1

    def close(self, outcome: str = "") -> None:
        if outcome:
            self.outcome = outcome
        for writer in self.writers.values():
            writer.close()
        if self.frames_file is not None:
            self.frames_file.close()
            self.frames_file = None
        ended_at = datetime.now(timezone.utc)
        elapsed_s = monotonic() - self.start_monotonic
        metadata = {
            "started_at_utc": self.started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "elapsed_s": elapsed_s,
            "frame_count": self.frame_index,
            "achieved_fps": self.frame_index / elapsed_s if elapsed_s > 0 else 0.0,
            "writer_fps": self.config.fps,
            "writer_codec": self.config.codec,
            "views": sorted(self.views),
            "videos": {
                view: str(self.output_dir / f"{view}{self.config.suffix}")
                for view in sorted(self.views)
            },
            "frames_csv": str(self.output_dir / "frames.csv"),
            "outcome": self.outcome,
            "platform": platform.platform(),
            "opencv_version": cv2.__version__,
            "extra": self.config.metadata or {},
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(make_json_safe(metadata), indent=2) + "\n",
            encoding="utf-8",
        )

    def __exit__(self, *_exc: object) -> None:
        self.close()
