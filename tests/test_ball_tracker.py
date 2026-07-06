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
