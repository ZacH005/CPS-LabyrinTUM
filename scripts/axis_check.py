#!/usr/bin/env python3
"""Measure which servo axis moves the ball along which board axis.

Procedure (the script walks you through it):
  1. Board level, ball placed in an OPEN area (no walls nearby).
  2. For each of 4 pulses (+yaw, -yaw, +pitch, -pitch):
       - you re-center the ball and press SPACE
       - the script measures the ball, applies a short tilt pulse,
         re-levels, and measures the displacement in board mm.
  3. It derives the command->board-axis mapping (handles swapped channels
     and reversed directions) and saves it to calibration/axis_map.npz.

The autonomous runner loads this file so the controller's board-frame
commands are routed to the right servo with the right sign.

Keys during the run: SPACE = start next pulse, q/Esc = abort.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.control.axis_map import snap_response_to_axis_map
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand
from cps_maze.vision.ball_pipeline import make_tracker

WINDOW = "axis check"
PULSES = [
    ("+yaw", np.array([1.0, 0.0])),
    ("-yaw", np.array([-1.0, 0.0])),
    ("+pitch", np.array([0.0, 1.0])),
    ("-pitch", np.array([0.0, -1.0])),
]


def measure_ball_mm(
    camera: CameraCapture,
    tracker,
    homography: Homography,
    samples: int,
    timeout_s: float = 5.0,
) -> np.ndarray:
    """Average ball position over `samples` found frames; raises on timeout."""
    positions = []
    deadline = time.monotonic() + timeout_s
    while len(positions) < samples:
        if time.monotonic() > deadline:
            raise RuntimeError("ball not detected - check lighting/tracker config")
        frame = camera.read()
        det = tracker.detect(frame.image)
        view = tracker.draw_detection(frame.image, det)
        cv2.imshow(WINDOW, view)
        cv2.waitKey(1)
        if det.found and det.x_px is not None and det.y_px is not None:
            positions.append(homography.image_point_to_board_mm(det.x_px, det.y_px))
    return np.mean(np.array(positions, dtype=float), axis=0)


def stream_command(
    link: ArduinoServoLink,
    yaw: float,
    pitch: float,
    seconds: float,
    camera: CameraCapture,
    tracker,
) -> None:
    """Stream a constant command while keeping the tracker fed with frames,
    so the track survives the pulse instead of losing the moving ball."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        link.send(ServoCommand(yaw=yaw, pitch=pitch))
        frame = camera.read()
        det = tracker.detect(frame.image)
        cv2.imshow(WINDOW, tracker.draw_detection(frame.image, det))
        cv2.waitKey(1)


def live_wait_for_space(
    camera: CameraCapture,
    tracker,
    mouse_state: dict,
    prompt: str,
) -> bool:
    """Live view while waiting: shows detection, the peak brightness under
    the mouse cursor (to pick min_specular), and lets the user click the
    ball to seed the tracker. Returns False if the user aborted."""
    while True:
        frame = camera.read()
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

        if mouse_state.pop("seed_request", None) is not None and hasattr(tracker, "seed"):
            sx, sy = mouse_state["last_click"]
            tracker.seed(sx, sy)
            print(f"seeded tracker at ({sx}, {sy})")

        det = tracker.detect(frame.image)
        view = tracker.draw_detection(frame.image, det)

        mx, my = mouse_state.get("pos", (0, 0))
        h, w = gray.shape
        x0, x1 = max(0, mx - 20), min(w, mx + 21)
        y0, y1 = max(0, my - 20), min(h, my + 21)
        peak = int(gray[y0:y1, x0:x1].max()) if (x1 > x0 and y1 > y0) else 0

        cv2.putText(view, prompt, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(view, f"peak brightness near cursor: {peak}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(view, "CLICK THE BALL to seed  |  SPACE=go  q=abort", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            return True
        if key in (27, ord("q")):
            return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--output", default="calibration/axis_map.npz")
    parser.add_argument("--port", default=None, help="Override serial port, e.g. COM10")
    parser.add_argument("--amplitude", type=float, default=0.25,
                        help="Starting pulse amplitude (0-1)")
    parser.add_argument("--max-amplitude", type=float, default=0.7,
                        help="Escalation cap when the ball refuses to move")
    parser.add_argument("--min-move-mm", type=float, default=5.0,
                        help="A pulse must move the ball at least this far to count")
    parser.add_argument("--pulse-seconds", type=float, default=0.6)
    parser.add_argument("--settle-seconds", type=float, default=1.5,
                        help="Wait after re-leveling before measuring")
    parser.add_argument("--samples", type=int, default=5,
                        help="Frames averaged per position measurement")
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)
    tracker = make_tracker(config.vision)
    port = args.port or config.serial["port"]

    cv2.namedWindow(WINDOW)
    mouse_state: dict = {"pos": (0, 0)}

    def on_mouse(event: int, x: int, y: int, *_rest) -> None:
        mouse_state["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state["last_click"] = (x, y)
            mouse_state["seed_request"] = True

    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)
    displacements: dict[str, np.ndarray] = {}

    with CameraCapture(config.camera) as camera, ArduinoServoLink(
        port=port,
        baudrate=int(config.serial["baudrate"]),
        timeout_s=float(config.serial["timeout_s"]),
    ) as link:
        time.sleep(2.0)  # Arduino reset after port open
        link.neutral()

        for name, direction in PULSES:
            amplitude = args.amplitude
            while True:
                prompt = f"[{name}] amplitude {amplitude:.2f}: ball in open area"
                print(f"\n{prompt} - click the ball, then SPACE (q aborts)")
                if not live_wait_for_space(camera, tracker, mouse_state, prompt):
                    print("aborted")
                    return

                link.neutral()
                time.sleep(args.settle_seconds)
                p0 = measure_ball_mm(camera, tracker, homography, args.samples)

                yaw, pitch = amplitude * direction
                stream_command(link, float(yaw), float(pitch), args.pulse_seconds,
                               camera, tracker)
                link.neutral()
                time.sleep(args.settle_seconds)
                p1 = measure_ball_mm(camera, tracker, homography, args.samples)

                moved = float(np.linalg.norm(p1 - p0))
                print(f"[{name}] displacement: dx={p1[0]-p0[0]:+.1f} mm, "
                      f"dy={p1[1]-p0[1]:+.1f} mm  (|{moved:.1f}| mm)")

                if moved >= args.min_move_mm:
                    # normalize by the amplitude that actually produced the move
                    displacements[name] = (p1 - p0) / amplitude
                    break
                if amplitude >= args.max_amplitude:
                    print(f"[{name}] still under {args.min_move_mm} mm at the "
                          f"amplitude cap - check this axis's linkage (loose horn "
                          f"screw? slack rod?). Retrying at the cap.")
                else:
                    amplitude = min(amplitude * 1.6, args.max_amplitude)
                    print(f"[{name}] too small to trust - escalating amplitude "
                          f"to {amplitude:.2f} and retrying")

        link.neutral()

    cv2.destroyAllWindows()

    # Response matrix: columns = board displacement per unit +yaw / +pitch.
    # Displacements are already normalized by the amplitude each pulse used;
    # (d+ - d-)/2 cancels board-tilt bias and gravity drift.
    yaw_col = (displacements["+yaw"] - displacements["-yaw"]) / 2.0
    pitch_col = (displacements["+pitch"] - displacements["-pitch"]) / 2.0
    response = np.column_stack([yaw_col, pitch_col])

    print("\nresponse matrix (board mm per unit command):")
    print(response.round(1))

    axis_map = snap_response_to_axis_map(response)
    axis_map.save(args.output)
    print(f"\nsaved axis map -> {args.output}")
    print("board->servo matrix:")
    print(axis_map.matrix)
    print("\nSanity check: a +x board command should now roll the ball toward "
          "+x (right in the calibrated frame). Verify in the first closed-loop "
          "test with a low --max-command.")


if __name__ == "__main__":
    main()
