"""
Final ball-tracking pipeline for the tilting labyrinth rig.

Two independent pieces, usable together or alone:

1. BallTracker -- motion + specular-highlight tracking. No trained model.
   At this camera's resolution the numbered holes look almost identical to
   the ball; the two reliable discriminators are motion (the ball moves,
   holes don't) and specular highlight (the ball hits ~254 brightness,
   holes/text top out ~150-215). Frame-to-frame differencing (not a fixed
   background) keeps this robust while the board tilts and its shading
   shifts.

2. BoardRectifier -- ArUco-marker board stabilization. Detects 4 corner
   markers and warps each frame to a fixed top-down canonical view, so
   downstream tracking never has to deal with perspective/tilt drift.
   Point it at footage with real printed markers on the board; it makes no
   assumption about how the markers got there.

Two-step workflow (calibration vs. streaming inference):
    Static-confuser detection needs to scan hundreds of frames spread
    across an entire recording to tell "always a bit bright" apart from
    "the ball, passing through" -- only possible with a complete recorded
    video, never with a live stream. So it's a separate, offline, run-once
    step; tracking itself only ever looks at the current + previous frame
    and loads the saved result, which is exactly what a live feed can do.

    # 1) calibrate once against footage you already have in full
    python pipeline.py --calibrate video.mp4 --confusers-file confusers.json

    # 2) track (this path never scans ahead -- fine for a live stream)
    python pipeline.py video.mp4 --seed-x 584 --seed-y 58 \\
        --confusers-file confusers.json \\
        --out-video annotated.mp4 --out-csv track.csv

    # same, but rectify each frame to a top-down view first (needs 4
    # ArUco markers -- real, physically printed on the board -- visible
    # in every frame)
    python pipeline.py video.mp4 --seed-x 584 --seed-y 58 --use-aruco \\
        --confusers-file wconfusers.json --out-video annotated.mp4

Known limitation of the classical tracker (inherent to motion-based
detection, not a bug): it cannot see the ball while it is perfectly still
(zero motion -> zero diff signal). Gaps are bridged with constant-velocity
prediction ("predicted" in the output) rather than left blank.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Detection core lives in the package so the live tools (axis_check,
# run_autonomous) run the exact same tracker; this CLI is the offline wrapper.
from cps_maze.vision.ball_pipeline import (
    BallTracker,
    auto_seed,
    load_calibration,
)

# --------------------------------------------------------------------------
# Board stabilization (ArUco)
# --------------------------------------------------------------------------

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# corner order is fixed: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
CORNER_IDS = [0, 1, 2, 3]


class BoardRectifier:
    """
    Detects the 4 corner markers and warps the frame to a fixed top-down
    canonical view. Requires 4 physical ArUco markers (dictionary
    DICT_4X4_50, ids 0-3) at the board's corners in that order.
    """

    def __init__(self, canonical_size=(700, 460)):
        self.canonical_w, self.canonical_h = canonical_size
        self.dst_pts = np.array([
            [0, 0],
            [self.canonical_w - 1, 0],
            [self.canonical_w - 1, self.canonical_h - 1],
            [0, self.canonical_h - 1],
        ], dtype=np.float32)
        self.detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

    def detect_corners(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None
        found = {}
        for c, i in zip(corners, ids.flatten()):
            if i in CORNER_IDS:
                found[i] = c.reshape(4, 2).mean(axis=0)  # marker center
        if len(found) < 4:
            return None
        return np.array([found[i] for i in CORNER_IDS], dtype=np.float32)

    def warp(self, frame):
        """Returns (warped_frame_or_None, homography_or_None, found_bool)."""
        src_pts = self.detect_corners(frame)
        if src_pts is None:
            return None, None, False
        H = cv2.getPerspectiveTransform(src_pts, self.dst_pts)
        warped = cv2.warpPerspective(frame, H, (self.canonical_w, self.canonical_h))
        return warped, H, True


# --------------------------------------------------------------------------
# Offline calibration (needs the full video; run once, not on a live feed)
# --------------------------------------------------------------------------

def compute_static_confusers(video_path, n_samples=200, thresh=225, freq_thresh=0.10, margin=18):
    """
    Find every board location that is bright enough to pass the ball's
    specular test *suspiciously often across the video* -- these are
    static reflective features (peg/hole rims), not the ball.

    Per pixel, count what fraction of sampled frames cross the specular
    threshold there. The real ball, traveling across the whole board over
    the video, cannot linger on any single pixel for more than a few
    percent of frames (verified: ~4% at its own start point, vs 35-93% for
    genuine confusers). A frequency threshold cleanly separates the two.
    """
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, n - 1), n_samples).astype(int)

    freq = None
    count = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        bright = (gray >= thresh)
        freq = bright.astype(np.uint32) if freq is None else freq + bright
        count += 1
    cap.release()
    if freq is None or count == 0:
        return []

    freq_map = (freq / count) >= freq_thresh
    mask = (freq_map.astype(np.uint8)) * 255
    n_comp, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Adjacent confusers can merge into one much larger blob. Size from the
    # bounding-box diagonal (correct for large/irregular merged shapes,
    # unlike area-derived radius) and only cap the resulting radius, so
    # oversized clusters still get excluded instead of silently dropped.
    max_r = 70
    out = []
    for i in range(1, n_comp):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        r = min(0.5 * float(np.hypot(w, h)), max_r)
        out.append((float(cx), float(cy), r + margin))
    return out


def parse_roi_arg(value):
    return [[float(v) for v in pair.split(",")] for pair in value.split(";")]


def load_roi_file(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    if "roi" not in data:
        raise ValueError(f"{path} does not contain an roi field")
    return data["roi"]


def video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    metadata = {
        "source_video": str(video_path),
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return metadata


def save_calibration(confusers, roi, path, metadata=None):
    payload = {"confusers": confusers, "roi": roi}
    if metadata:
        payload["metadata"] = metadata
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _read_rectified(cap, rectifier):
    """Read one frame, rectify it if a BoardRectifier is given. Returns
    (ok, frame_to_use) -- frame_to_use is None if rectification failed."""
    ok, frame = cap.read()
    if not ok:
        return False, None
    if rectifier is None:
        return True, frame
    warped, _, found = rectifier.warp(frame)
    return True, (warped if found else None)


def main():
    ap = argparse.ArgumentParser(description="Track the metallic ball through rig video.")
    ap.add_argument("video", nargs="?", help="video to track (not required for --calibrate)")
    ap.add_argument("--seed-x", type=float)
    ap.add_argument("--seed-y", type=float)
    ap.add_argument("--seed-r", type=float, default=9)
    ap.add_argument("--auto-seed", action="store_true", help="find seed from the first frame automatically")
    ap.add_argument("--min-specular", type=int, default=225,
                    help="brightness gate for specular candidates")
    ap.add_argument("--max-jump-px", type=float, default=35,
                    help="candidate gate around predicted position")
    ap.add_argument("--max-search-px", type=float, default=60,
                    help="maximum expanded candidate search radius")
    ap.add_argument("--search-growth", type=float, default=1.15,
                    help="search-radius growth per missed frame")
    ap.add_argument("--max-predict-frames", type=int, default=8,
                    help="number of frames to coast before reporting lost")
    ap.add_argument("--max-velocity-px-per-frame", type=float, default=35,
                    help="prediction velocity clamp; <=0 disables it")
    ap.add_argument("--max-single-frame-jump-px", type=float, default=45,
                    help="hard gate from last accepted position; <=0 disables it")
    ap.add_argument("--disable-global-reacquire", action="store_true",
                    help="after long loss, do not reacquire from any bright spot")
    ap.add_argument("--out-video", default=None, help="optional annotated output video")
    ap.add_argument("--out-csv", default="track.csv")
    ap.add_argument("--end-frame", type=int, default=None,
                     help="stop before this frame index (use to cut trailing non-maze footage)")
    ap.add_argument("--start-frame", type=int, default=0,
                     help="seed position is read from this frame (use when the ball isn't visible at frame 0)")
    ap.add_argument("--calibrate", metavar="SOURCE_VIDEO",
                     help="offline step: scan a full recorded video for static confusers "
                          "(holes/pegs that glint like the ball) and save them -- run this "
                          "once against footage you already have in full. Record the video "
                          "with NO ball on the board: a ball lingering anywhere for >10%% "
                          "of the recording blacklists its own resting spots and the live "
                          "tracker then loses it there. Not usable on a live stream, which "
                          "by definition has no 'whole video' to scan.")
    ap.add_argument("--confuser-thresh", type=int, default=225,
                    help="brightness threshold used by --calibrate")
    ap.add_argument("--confuser-freq-thresh", type=float, default=0.10,
                    help="minimum bright-frame frequency for --calibrate confusers")
    ap.add_argument("--confuser-margin", type=float, default=18,
                    help="extra pixel radius around each --calibrate confuser")
    ap.add_argument("--confusers-file", default="confusers.json",
                     help="where --calibrate saves to / normal tracking loads from")
    ap.add_argument("--roi", default=None,
                     help="playable-board polygon as 'x1,y1;x2,y2;...' (only used with --calibrate); "
                          "candidates outside it are never the ball, regardless of brightness/motion")
    ap.add_argument("--roi-file", default=None,
                     help="JSON file containing an roi field; alternative to --roi for --calibrate")
    ap.add_argument("--use-aruco", action="store_true",
                     help="rectify each frame to a top-down view via 4 ArUco corner markers "
                          "(DICT_4X4_50, ids 0-3) before doing anything else -- needs real "
                          "markers physically on the board, visible in every frame")
    ap.add_argument("--canonical-size", default="700,460",
                     help="canonical warped frame size 'w,h', only used with --use-aruco")
    args = ap.parse_args()

    rectifier = None
    if args.use_aruco:
        cw, ch = (int(v) for v in args.canonical_size.split(","))
        rectifier = BoardRectifier(canonical_size=(cw, ch))

    if args.calibrate:
        if args.roi and args.roi_file:
            raise SystemExit("pass either --roi or --roi-file, not both")
        print(f"scanning {args.calibrate} for static confusers (bright pegs/holes)...")
        confusers = compute_static_confusers(
            args.calibrate,
            thresh=args.confuser_thresh,
            freq_thresh=args.confuser_freq_thresh,
            margin=args.confuser_margin,
        )
        roi = None
        roi_source = None
        if args.roi_file:
            roi = load_roi_file(args.roi_file)
            roi_source = str(args.roi_file)
        elif args.roi:
            roi = parse_roi_arg(args.roi)
            roi_source = "--roi"
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "roi_source": roi_source,
            **video_metadata(args.calibrate),
        }
        save_calibration(confusers, roi, args.confusers_file, metadata=metadata)
        print(f"found {len(confusers)} confuser(s)" + (f", roi with {len(roi)} points" if roi else ", no roi given")
              + f", saved to {args.confusers_file}")
        if len(confusers) > 30:
            print("WARNING: that is far more confusers than a board has "
                  "static bright features (~5-15). The BALL was almost "
                  "certainly in this video and has blacklisted its own "
                  "resting spots - the live tracker will then lose it there. "
                  "Re-record with NO ball on the board and calibrate again.")
        return

    if not args.video:
        raise SystemExit("pass a video to track, or --calibrate SOURCE_VIDEO to (re)build confusers first")

    cap = cv2.VideoCapture(args.video)
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ok, frame = _read_rectified(cap, rectifier)
    if not ok:
        raise SystemExit("could not read video")
    if frame is None:
        raise SystemExit("--use-aruco: could not find all 4 corner markers on the seed frame")
    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if args.auto_seed:
        seed = auto_seed(gray0, args.min_specular)
        if seed is None:
            raise SystemExit("auto-seed failed: no bright specular blob in the seed frame, pass --seed-x/--seed-y")
        print(f"auto-seed: {seed}")
    else:
        if args.seed_x is None or args.seed_y is None:
            raise SystemExit("pass --seed-x/--seed-y or --auto-seed")
        seed = (args.seed_x, args.seed_y)

    # Streaming-safe: this loads a file written once by a prior --calibrate
    # pass over footage we already had in full. It does NOT scan args.video
    # itself, so this same code path works unchanged on a live feed -- only
    # the calibration step needs a complete recording.
    confusers, roi = [], None
    if Path(args.confusers_file).exists():
        confusers, roi = load_calibration(args.confusers_file)
        print(f"loaded {len(confusers)} confuser(s)" + (f" and roi ({len(roi)} pts)" if roi else " (no roi)")
              + f" from {args.confusers_file}")
    else:
        print(f"no confusers file at {args.confusers_file} -- run with --calibrate first; "
              f"tracking without confuser/roi exclusion for now")

    tracker = BallTracker(
        seed,
        seed_r=args.seed_r,
        max_jump=args.max_jump_px,
        max_search=args.max_search_px,
        search_growth=args.search_growth,
        min_specular=args.min_specular,
        max_predict_frames=args.max_predict_frames,
        static_confusers=confusers,
        roi=roi,
        max_velocity_px_per_frame=args.max_velocity_px_per_frame,
        max_single_frame_jump_px=args.max_single_frame_jump_px,
        allow_global_reacquire=not args.disable_global_reacquire,
    )

    writer = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))

    rows = []
    counts = {"seed": 0, "detected": 0, "predicted": 0, "lost": 0}

    idx = args.start_frame
    while True:
        if args.end_frame is not None and idx >= args.end_frame:
            break
        if frame is None:
            # --use-aruco: markers not found this frame -- skip tracking,
            # keep going (never blocks on a single bad frame in a stream)
            idx += 1
            ok, frame = _read_rectified(cap, rectifier)
            if not ok:
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, r, status = tracker.update(gray)
        counts[status] += 1
        rows.append((idx, round(x, 1), round(y, 1), round(r, 1), status))

        if writer is not None:
            color = {"seed": (255, 0, 0), "detected": (0, 255, 0),
                     "predicted": (0, 200, 255), "lost": (0, 0, 255)}[status]
            out = frame.copy()
            pad = r * 1.3
            x0, y0 = int(x - pad), int(y - pad)
            x1, y1 = int(x + pad), int(y + pad)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            cv2.putText(out, status, (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            writer.write(out)

        idx += 1
        ok, frame = _read_rectified(cap, rectifier)
        if not ok:
            break

    cap.release()
    if writer is not None:
        writer.release()

    with open(args.out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "x", "y", "r", "status"])
        wr.writerows(rows)

    total = sum(counts.values())
    print(f"frames: {total}")
    for k, v in counts.items():
        print(f"  {k}: {v} ({100*v/total:.1f}%)" if total else f"  {k}: {v}")
    print(f"wrote {args.out_csv}" + (f" and {args.out_video}" if args.out_video else ""))


if __name__ == "__main__":
    main()
