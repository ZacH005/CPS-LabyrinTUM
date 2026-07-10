#!/usr/bin/env python3
"""Ball tracking debugger: see exactly WHY and WHEN the tracker loses the ball.

Live view with per-frame internals of the pipeline tracker:
- every candidate the tracker considered (orange = motion cue, cyan = specular
  cue) and how many were rejected at each gate (ROI/confuser, jump, specular)
- the glint brightness at the tracked position vs the min_specular gate
- template match score and whether the template rescued the track
- status transitions logged with timestamps and the dominant rejection reason

The min_specular trackbar changes the gate LIVE, so the threshold can be
calibrated against reality: roll the ball everywhere it will travel (including
the dimmest corners) and watch the glint value; the exit report recommends a
threshold from the recorded data.

Controls:
  click        : seed the tracker on the ball
  min_specular : trackbar, live gate adjustment
  p            : pause/resume the live view
  q/Esc        : quit and print the report

A per-frame CSV is written for offline analysis.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.vision.ball_pipeline import PipelineBallTracker

WINDOW = "tracking debug"


def dominant_loss_reason(dbg: dict) -> str:
    """Best explanation for a frame with no accepted candidate."""
    if dbg.get("n_motion", 0) + dbg.get("n_highlight", 0) == 0:
        return "no candidates at all (glint below gate AND no motion signal)"
    if dbg.get("n_rej_specular", 0) > 0:
        return "candidates failed the specular gate (min_specular too high?)"
    if dbg.get("n_rej_jump", 0) > 0:
        return "candidates rejected as implausible jumps (search window too tight?)"
    if dbg.get("n_rej_roi_confuser", 0) > 0:
        return "candidates removed by ROI/confuser filtering"
    return "candidates existed but none matched (mixed gates)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--log", default="data/raw/tracking_debug.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    tracker = PipelineBallTracker(config.vision)
    initial_gate = tracker.min_specular

    # Homography is NOT needed for tracking (pixel space), but drawing the
    # board frame verifies the calibration at a glance: the magenta outline
    # must hug the play area. If it does not, recalibrate corners and
    # regenerate the derived artifacts (path/holes/walls AND roi/confusers).
    homography = None
    board_px = None
    if Path(args.homography).exists():
        homography = Homography.load(args.homography)
        try:
            bw = float(config.maze["width_mm"])
            bh = float(config.maze["height_mm"])
            border = np.array([[0.0, 0.0], [bw, 0.0], [bw, bh], [0.0, bh]])
            board_px = homography.board_points_to_image_px(border).astype(np.int32)
        except (KeyError, TypeError, ValueError):
            pass
    else:
        print(f"note: no homography at {args.homography} - board overlay and "
              "mm readout disabled (tracking itself does not need it)")

    mouse: dict = {}

    def on_mouse(event: int, x: int, y: int, *_rest) -> None:
        mouse["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["seed"] = (x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cv2.createTrackbar("min_specular", WINDOW, initial_gate, 255, lambda _v: None)
    print(__doc__)

    glints_while_tracked: list[int] = []
    loss_events: list[tuple[float, str]] = []
    status_counts = {"detected": 0, "predicted": 0, "lost": 0, "unseeded": 0}
    prev_status = None
    paused = False
    t0 = time.monotonic()
    loop_dts: list[float] = []
    jumps_px: list[float] = []
    prev_ball = None
    prev_t = time.monotonic()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with CameraCapture(config.camera) as camera, log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_s", "status", "found", "x_px", "y_px", "glint_at_track",
            "min_specular", "n_motion", "n_highlight", "n_rej_roi_confuser",
            "n_rej_jump", "n_rej_specular", "template_score",
            "template_rescued", "miss_streak",
        ])

        while True:
            if not paused:
                frame = camera.read()
                image = frame.image

            gate = cv2.getTrackbarPos("min_specular", WINDOW)
            if gate != tracker.min_specular:
                tracker.min_specular = gate
                if tracker.tracker is not None:
                    tracker.tracker.min_specular = gate

            seed = mouse.pop("seed", None)
            if seed is not None:
                tracker.seed(*seed)
                print(f"seeded at {seed}")

            detection = tracker.detect(image)
            now = time.monotonic()
            if not paused:
                loop_dts.append(now - prev_t)
            prev_t = now
            if detection.found:
                if prev_ball is not None:
                    jumps_px.append(float(np.hypot(
                        detection.x_px - prev_ball[0],
                        detection.y_px - prev_ball[1])))
                prev_ball = (detection.x_px, detection.y_px)
            else:
                prev_ball = None
            dbg = dict(getattr(tracker.tracker, "debug", {}) or {}) \
                if tracker.tracker is not None else {}
            status = dbg.get("status", "unseeded")
            status_counts[status] = status_counts.get(status, 0) + 1
            t = time.monotonic() - t0

            glint = int(dbg.get("peak_at_track", 0))
            if status == "detected":
                glints_while_tracked.append(glint)

            # log every status transition into a loss with its reason
            if status in ("predicted", "lost") and prev_status == "detected":
                reason = dominant_loss_reason(dbg)
                loss_events.append((t, reason))
                print(f"[{t:7.1f}s] track dropped: {reason} "
                      f"(glint at track {glint}, gate {gate})")
            prev_status = status

            writer.writerow([
                f"{t:.3f}", status, detection.found,
                f"{detection.x_px:.1f}" if detection.found else "",
                f"{detection.y_px:.1f}" if detection.found else "",
                glint, gate,
                dbg.get("n_motion", ""), dbg.get("n_highlight", ""),
                dbg.get("n_rej_roi_confuser", ""), dbg.get("n_rej_jump", ""),
                dbg.get("n_rej_specular", ""),
                f"{tracker.last_template_score:.2f}"
                if not np.isnan(tracker.last_template_score) else "",
                tracker.template_rescued, dbg.get("miss_streak", ""),
            ])

            # ---- draw ----
            view = image.copy()
            if board_px is not None:
                # magenta = calibrated board frame; must hug the play area
                cv2.polylines(view, [board_px], True, (255, 0, 255), 2)
            for cand in dbg.get("candidates", []):
                cx, cy, cr = cand[0], cand[1], cand[2]
                source = cand[3] if len(cand) > 3 else "motion"
                color = (0, 165, 255) if source == "motion" else (255, 255, 0)
                cv2.circle(view, (int(cx), int(cy)), max(int(cr), 3), color, 1)
            if tracker.roi:
                pts = np.array(tracker.roi, dtype=np.int32)
                cv2.polylines(view, [pts], True, (255, 255, 0), 1)
            for (sx, sy, sr) in tracker.confusers:
                cv2.circle(view, (int(sx), int(sy)), int(sr), (0, 0, 255), 1)
            if detection.found:
                c = (int(detection.x_px), int(detection.y_px))
                cv2.circle(view, c, int(detection.radius_px or 6), (0, 255, 0), 2)

            color = {"detected": (0, 255, 0), "predicted": (0, 200, 255),
                     "lost": (0, 0, 255)}.get(status, (200, 200, 200))
            recent = loop_dts[-30:]
            loop_fps = (len(recent) / max(sum(recent), 1e-9)) if recent else 0.0
            last_jump = jumps_px[-1] if jumps_px else 0.0
            lines = [
                f"status: {status}   miss streak: {dbg.get('miss_streak', '-')}"
                f"   loop: {loop_fps:.0f} fps   jump: {last_jump:.0f} px/frame",
                f"glint at track: {glint}   gate (min_specular): {gate}",
                f"candidates: motion {dbg.get('n_motion', '-')} + "
                f"highlight {dbg.get('n_highlight', '-')}  |  rejected: "
                f"roi/conf {dbg.get('n_rej_roi_confuser', '-')}  "
                f"jump {dbg.get('n_rej_jump', '-')}  "
                f"specular {dbg.get('n_rej_specular', '-')}",
                f"template score: "
                + (f"{tracker.last_template_score:.2f}"
                   if not np.isnan(tracker.last_template_score) else "-")
                + ("  [RESCUED TRACK]" if tracker.template_rescued else ""),
                "click ball = seed   p = pause   q = quit + report",
            ]
            for k, text in enumerate(lines):
                cv2.putText(view, text, (10, 28 + 26 * k),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            color if k == 0 else (0, 255, 255), 2)
            if "pos" in mouse:
                mx, my = mouse["pos"]
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                x0, x1 = max(0, mx - 15), min(w, mx + 16)
                y0, y1 = max(0, my - 15), min(h, my + 16)
                if x1 > x0 and y1 > y0:
                    cursor = f"cursor peak: {int(gray[y0:y1, x0:x1].max())}"
                    if homography is not None:
                        bx, by = homography.image_point_to_board_mm(mx, my)
                        cursor += f"   board: {bx:.0f}, {by:.0f} mm"
                    cv2.putText(view, cursor, (10, view.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                paused = not paused

    cv2.destroyAllWindows()

    # ---- report ----
    total = sum(status_counts.values())
    print("\n===== tracking debug report =====")
    print(f"frames: {total}   log: {log_path}")
    if loop_dts:
        fps = 1.0 / max(float(np.median(loop_dts)), 1e-9)
        print(f"processing loop: median {fps:.0f} fps "
              f"(camera mode is irrelevant if the loop is slower)")
    if jumps_px:
        j = np.array(jumps_px)
        print(f"ball movement between processed frames: median "
              f"{np.median(j):.0f} px, p95 {np.percentile(j, 95):.0f} px, "
              f"max {j.max():.0f} px")
        print(f"  -> jump gates (max_jump_px etc.) must comfortably exceed "
              f"the p95; current max_jump_px is "
              f"{tracker.max_jump_px:.0f}")
    for k, v in status_counts.items():
        if v:
            print(f"  {k}: {v} ({100 * v / max(total, 1):.0f}%)")
    if loss_events:
        print(f"\ntrack drops: {len(loss_events)}")
        reasons: dict[str, int] = {}
        for _, r in loss_events:
            reasons[r] = reasons.get(r, 0) + 1
        for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n}x  {r}")
    if glints_while_tracked:
        g = np.array(glints_while_tracked)
        p5, p50 = np.percentile(g, 5), np.percentile(g, 50)
        print(f"\nball glint while tracked: p5={p5:.0f}  median={p50:.0f}  "
              f"min={g.min()}  max={g.max()}")
        rec = max(int(p5) - 6, 150)
        print(f"RECOMMENDED vision.min_specular: ~{rec} "
              f"(p5 of the ball's real glint minus margin; current gate "
              f"{tracker.min_specular})")
        print("Roll the ball through the DIMMEST part of the maze during the "
              "session or this recommendation will be too optimistic.")
    else:
        print("\nno tracked frames recorded - seed the ball and try again")


if __name__ == "__main__":
    main()
