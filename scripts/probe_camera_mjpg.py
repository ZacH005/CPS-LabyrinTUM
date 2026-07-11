#!/usr/bin/env python3
"""Decide whether this camera can stream MJPG (compressed) at full resolution.

The fps probe showed the camera stays on uncompressed YUY2 at 1280x800 (=> ~10
fps, USB2 bandwidth limited) and only reaches 30 fps at 640x480. MJPG would
compress each frame ~10:1 and let 1280x800 run fast over USB2 WITHOUT changing
resolution (so no recalibration). But OpenCV's MJPG negotiation is famously
sensitive to backend and to the ORDER of the set() calls, so a single failed
attempt does not prove the camera lacks MJPG.

This tries a matrix of (backend x set-order x resolution) and reports, for each,
the fourcc the driver actually negotiated and the MEASURED fps. Interpretation:

  * If ANY row at 1280x800 shows fourcc=MJPG and fast fps  -> we can fix the
    frame rate with a camera.py change and NO recalibration. Tell me which row.
  * If MJPG only negotiates at 640x480 (or never)          -> the camera cannot
    do MJPG at full res; we drop to 640x480 and recalibrate.

Report-only: nothing is written. Run at the lab and paste the whole output.
Note: the MSMF backend can take 10-30 s to OPEN each camera - be patient; that
open cost is one-time and irrelevant to the run loop's frame rate.
"""
from __future__ import annotations

import argparse
import sys
from time import perf_counter

import cv2


def _decode_fourcc(value: float) -> str:
    code = int(value)
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars if all(32 <= ord(c) <= 126 for c in chars) else str(code)


def _measure_fps(cap: cv2.VideoCapture, seconds: float) -> float:
    for _ in range(5):
        cap.read()
    frames = 0
    start = perf_counter()
    while perf_counter() - start < seconds:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    elapsed = perf_counter() - start
    return frames / elapsed if elapsed > 0 else 0.0


def _try(device: int, backend: int, order: str, w: int, h: int,
         seconds: float) -> tuple[str, float, int, int]:
    cap = cv2.VideoCapture(device, backend)
    if not cap.isOpened():
        return ("open-failed", 0.0, 0, 0)
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    if order == "fourcc_first":
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    elif order == "fourcc_last":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    elif order == "fourcc_last_read":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.read()  # some drivers only apply the mode after the first grab
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        cap.read()
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = _measure_fps(cap, seconds)
    cap.release()
    return (fourcc, fps, ow, oh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.5)
    args = parser.parse_args()

    backends: list[tuple[str, int]] = []
    if sys.platform == "win32":
        backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    else:
        backends = [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]

    orders = ["fourcc_first", "fourcc_last", "fourcc_last_read"]
    resolutions = [(1280, 800), (640, 480)]

    print(f"Probing device {args.device}. Looking for fourcc=MJPG at 1280x800.\n")
    print(f"{'backend':7} {'order':17} {'requested':10} {'-> negotiated':14} {'fps':>6}")
    print("-" * 62)
    win = []
    for bname, backend in backends:
        for w, h in resolutions:
            for order in orders:
                try:
                    fourcc, fps, ow, oh = _try(args.device, backend, order, w, h,
                                               args.seconds)
                except Exception as exc:
                    fourcc, fps, ow, oh = (f"err:{exc}", 0.0, 0, 0)
                got = f"{ow}x{oh} {fourcc}"
                flag = ""
                is_mjpg = fourcc.upper() in ("MJPG", "MJPEG", "MJG")
                if is_mjpg and fps >= 40:
                    flag = "  <== WIN"
                    win.append((bname, order, ow, oh, fps))
                elif fps >= 40:
                    flag = "  (fast)"
                print(f"{bname:7} {order:17} {w}x{h:<5} {got:14} {fps:6.1f}{flag}")
        print()

    print("=" * 62)
    if any(ow >= 1280 for _, _, ow, _, _ in win):
        print("MJPG WORKS at full resolution. We can fix fps with a camera.py")
        print("change and NO recalibration. Winning rows:")
        for b, o, ow, oh, fps in win:
            if ow >= 1280:
                print(f"   backend={b} order={o} {ow}x{oh} @ {fps:.0f} fps")
    elif win:
        print("MJPG only works below full resolution. Options: run at that")
        print("resolution and recalibrate, or accept it. Winning rows:")
        for b, o, ow, oh, fps in win:
            print(f"   backend={b} order={o} {ow}x{oh} @ {fps:.0f} fps")
    else:
        print("No MJPG mode negotiated on any backend/order. This camera cannot")
        print("stream compressed at these resolutions -> to go faster than 10 fps")
        print("we must drop to 640x480 YUY2 (30 fps) and recalibrate the chain.")


if __name__ == "__main__":
    main()
