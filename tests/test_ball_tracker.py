import cv2
import numpy as np

from cps_maze.vision.ball_pipeline import BallTracker, PipelineBallTracker
from cps_maze.vision.ball_tracker import BrightBlobBallTracker


def test_tracker_update_from_config_updates_runtime_values():
    tracker = BrightBlobBallTracker(
        {
            "min_blob_area_px": 25,
            "max_blob_area_px": 5000,
            "threshold_value": 220,
            "blur_kernel": 5,
            "morph_kernel": 5,
            "clahe_clip": 2.0,
            "smoothing_alpha": 0.6,
            "use_otsu": False,
            "debug_overlay": False,
            "debug_overlay_every_n": 1,
            "debug_start_stage": "binary",
        }
    )

    tracker.update_from_config(
        {
            "min_blob_area_px": 50,
            "max_blob_area_px": 2000,
            "threshold_value": 180,
            "blur_kernel": 7,
            "morph_kernel": 3,
            "clahe_clip": 1.5,
            "smoothing_alpha": 0.8,
            "use_otsu": True,
            "debug_overlay": True,
            "debug_overlay_every_n": 2,
            "debug_start_stage": "morph",
        }
    )

    assert tracker.min_area == 50.0
    assert tracker.max_area == 2000.0
    assert tracker.threshold_value == 180
    assert tracker.blur_kernel == 7
    assert tracker.morph_kernel == 3
    assert tracker.clahe_clip == 1.5
    assert tracker.smoothing_alpha == 0.8
    assert tracker.use_otsu is True
    assert tracker.debug_enabled is True
    assert tracker.debug_every_n == 2
    assert tracker.debug_start_stage == "morph"


def test_pipeline_tracker_rejects_implausible_far_bright_snap():
    tracker = BallTracker(
        (20, 20),
        min_specular=240,
        max_predict_frames=0,
        allow_global_reacquire=False,
    )
    blank = np.zeros((100, 100), dtype=np.uint8)
    assert tracker.update(blank)[3] == "seed"

    frame = blank.copy()
    cv2.circle(frame, (80, 80), 3, 255, -1)

    x, y, _r, status = tracker.update(frame)

    assert status == "lost"
    assert np.hypot(x - 80, y - 80) > 70


def test_manual_seed_overrides_static_confuser_at_seed():
    tracker = BallTracker(
        (20, 20),
        min_specular=240,
        static_confusers=[(20, 20, 30)],
        allow_global_reacquire=False,
    )
    blank = np.zeros((100, 100), dtype=np.uint8)
    frame = blank.copy()
    cv2.circle(frame, (24, 20), 3, 255, -1)

    assert tracker.update(blank)[3] == "seed"
    x, y, _r, status = tracker.update(frame)

    assert status == "detected"
    assert np.hypot(x - 24, y - 20) < 5


def test_live_pipeline_does_not_auto_seed_by_default():
    tracker = PipelineBallTracker({"min_specular": 240})
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(frame, (40, 40), 4, (255, 255, 255), -1)

    detection = tracker.detect(frame)

    assert detection.found is False


def test_live_pipeline_does_not_report_predicted_frames_by_default():
    tracker = PipelineBallTracker({"min_specular": 240, "max_predict_frames": 2})
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    tracker.detect(frame)
    tracker.seed(30, 30)
    seeded = tracker.detect(frame)
    predicted = tracker.detect(frame)

    assert seeded.found is True
    assert predicted.found is False


def test_moving_ball_with_blurred_glint_passes_motion_gate():
    # Motion blur dims the glint exactly when the ball moves fast: the
    # motion cue must accept a candidate whose peak is below the static
    # highlight gate but above the (lower) motion gate.
    tracker = BallTracker(
        (20, 20),
        min_specular=237,
        motion_min_specular=185,
        max_jump=60,
        max_single_frame_jump_px=90,
        allow_global_reacquire=False,
    )
    f0 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(f0, (20, 20), 4, 200, -1)  # blurred glint: 200 < 237
    assert tracker.update(f0)[3] == "seed"

    f1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(f1, (45, 20), 4, 200, -1)  # moved 25px, still dim

    x, _y, _r, status = tracker.update(f1)

    assert status == "detected"
    assert abs(x - 45) < 6
