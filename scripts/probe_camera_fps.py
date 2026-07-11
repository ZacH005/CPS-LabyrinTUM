#!/usr/bin/env python3
"""Measure the camera's REAL delivered frame rate and format.

Why this exists: the autonomous run's control loop was measured at 10.6 Hz
(94 ms per iteration) even though the tracker + drawing + serial together cost
under 10 ms. The missing ~85 ms is cv2.VideoCapture.read() BLOCKING while it
waits for the next frame - i.e. the camera is only DELIVERING ~10 fps, not the
120 fps it is capable of. A 10 Hz loop cannot stabilize a fast marble (the ball
moves ~9 mm between updates and the velocity estimate lags a full frame), so the
controller over-corrects and the ball rings. This is the root cause of the
"can't stabilize" behavior - not the control gains.

This probe opens the camera the same way the runner does, reports what format
and rate the driver actually negotiated, MEASURES the true delivered fps over a
few seconds, and then tries a handful of alternate configurations so you can see
immediately which one restores a fast frame rate. Nothing here is written to
config - it only reports. Run it at the lab and paste the output.

Usage:
    python scripts/probe_camera_fps.py                 # uses configs/*.yaml device
    python scripts/probe_camera_fps.py --device 1      # override device index
    python scripts/probe_camera_fps.py --seconds 4     # longer measurement
"""
from __future__ import annotations

import argparse
import sys
from time import perf_counter

import cv2

try:
    from cps_maze.config import load_config
except Exception:  # pragma: no cover - allow running without full package
    load_config = None


def _decode_fourcc(value: float) -> str:
    code = int(value)
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars if all(32 <= ord(c) <= 126 for c in chars) else str(code)


def _measure(cap: cv2.VideoCapture, seconds: float) -> tuple[float, int, float]:
    """Return (delivered_fps, frames, median_read_ms) over `seconds`."""
    # warm up - the first few reads after configuring are often slow
    for _ in range(5):
        cap.read()
    reads_ms: list[float] = []
    frames = 0
    start = perf_counter()
    while perf_counter() - start < seconds:
        t0 = perf_counter()
        ok, _img = cap.read()
        dt = (perf_counter() - t0) * 1000.0
        if not ok:
            break
        reads_ms.append(dt)
        frames += 1
    elapsed = perf_counter() - start
    reads_ms.sort()
    median_ms = reads_ms[len(reads_ms) // 2] if reads_ms else float("nan")
    fps = frames / elapsed if elapsed > 0 else 0.0
    return fps, frames, median_ms


def _open(device: int, width: int, height: int, fps: int,
          fourcc: str | None) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(device, backend)
    if not cap.isOpened():
        return cap
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _report(cap: cv2.VideoCapture, label: str, seconds: float) -> None:
    if not cap.isOpened():
        print(f"  [{label}] could not open device")
        return
    obs_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    obs_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    obs_cc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    obs_fps = cap.get(cv2.CAP_PROP_FPS)
    fps, frames, median_ms = _measure(cap, seconds)
    flag = "  <-- FAST" if fps >= 40 else ("  <-- slow" if fps < 20 else "")
    print(f"  [{label}] {obs_w}x{obs_h} fourcc={obs_cc} driver_fps={obs_fps:.0f}"
          f"  ->  MEASURED {fps:.1f} fps ({median_ms:.0f} ms/read){flag}")


def _exposure_report(cap: cv2.VideoCapture) -> None:
    # Values are driver-defined and often on a log2 scale for UVC; we only
    # want to know whether auto-exposure is on and roughly how long exposure is.
    ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    print(f"  auto_exposure={ae}  exposure={exp}  gain={gain}")
    print("  (long exposure in dim light drags fps down AND causes the motion")
    print("   blur that smears the ball's glint - a double win to fix.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", type=int, default=None,
                        help="override device index")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--via-runner", action="store_true",
                        help="measure through the real CameraCapture class "
                             "(the production path, honors the MSMF backend)")
    args = parser.parse_args()

    # --via-runner: measure through the ACTUAL CameraCapture class the runner
    # uses (honors camera.backend, now MSMF on Windows). This is the true
    # production path - if this reads fast, the run loop will too.
    if args.via_runner:
        from cps_maze.camera import CameraCapture
        cfg = load_config(args.config)
        if args.device is not None:
            cfg.camera["device_index"] = args.device
        print("Opening via CameraCapture (production path). MSMF can take up "
              "to ~30 s to open - be patient.")
        with CameraCapture(cfg.camera) as cam:
            obs = cam.observed_settings()
            print(f"  negotiated: {obs['width']}x{obs['height']} "
                  f"fourcc={obs['fourcc']} driver_fps={obs['fps']:.0f}")
            for _ in range(5):
                cam.read()
            frames = 0
            start = perf_counter()
            while perf_counter() - start < max(args.seconds, 3.0):
                cam.read()
                frames += 1
            elapsed = perf_counter() - start
            measured = frames / elapsed
            flag = "  <-- FAST, run loop fixed" if measured >= 40 else "  <-- still slow"
            print(f"  MEASURED {measured:.1f} fps through CameraCapture{flag}")
        return

    device = args.device
    width, height, fps = 1280, 800, 120
    if device is None and load_config is not None:
        try:
            cfg = load_config(args.config)
            device = int(cfg.camera["device_index"])
            width = int(cfg.camera["width"])
            height = int(cfg.camera["height"])
            fps = int(cfg.camera["fps"])
        except Exception as exc:
            print(f"(could not read config: {exc}; using device 0)")
    if device is None:
        device = 0

    print(f"Probing camera device {device} for {args.seconds:.0f}s each.\n")

    print(f"1) CURRENT runner config: {width}x{height} @ {fps} MJPG")
    cap = _open(device, width, height, fps, "MJPG")
    _report(cap, "current", args.seconds)
    if cap.isOpened():
        _exposure_report(cap)
    cap.release()

    # Alternates: keep MJPG but confirm it is honored; try uncompressed to
    # confirm the fallback penalty; try a smaller frame that USB2 can stream
    # fast even uncompressed. NOTE: changing WIDTH/HEIGHT invalidates the
    # homography and every calibration file - only drop resolution as a last
    # resort, and re-run the calibration chain if you do.
    print("\n2) Alternates (report only - see note before changing resolution):")
    for label, w, h, cc in [
        ("MJPG-720", 1280, 720, "MJPG"),
        ("MJPG-800x600", 800, 600, "MJPG"),
        ("MJPG-640", 640, 480, "MJPG"),
        ("YUY2-current", width, height, "YUY2"),
    ]:
        cap = _open(device, w, h, fps, cc)
        _report(cap, label, args.seconds)
        cap.release()

    print("\nRead the MEASURED column, not driver_fps (drivers lie).")
    print("If 'current' measures < 20 fps but an MJPG alternate is fast, the")
    print("camera is falling back to uncompressed at this resolution. If ALL")
    print("MJPG rows are slow, it is exposure - fix lighting / lock a short")
    print("exposure. Paste this whole output back.")


if __name__ == "__main__":
    main()
