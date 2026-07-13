import json

import numpy as np
import pytest

from cps_maze.logging.run_recorder import (
    RunRecordingConfig,
    RunVideoRecorder,
    parse_record_views,
)


def test_parse_record_views_accepts_comma_string():
    assert parse_record_views("raw, overlay, motion") == {"raw", "overlay", "motion"}


def test_parse_record_views_rejects_unknown_view():
    with pytest.raises(ValueError, match="unknown recording view"):
        parse_record_views("raw,banana")


def test_run_video_recorder_writes_sidecars(tmp_path):
    recorder = RunVideoRecorder(RunRecordingConfig(
        output_dir=tmp_path,
        views={"raw"},
        fps=10.0,
        codec="MJPG",
    ))
    image = np.zeros((24, 32, 3), dtype=np.uint8)

    with recorder:
        recorder.write(
            image,
            123.0,
            tracker=object(),
            armed=True,
            found=False,
            status="test frame",
        )
        recorder.set_outcome("test complete")

    assert (tmp_path / "raw.avi").exists()
    frames_csv = (tmp_path / "frames.csv").read_text(encoding="utf-8")
    assert "frame_index,timestamp_s,elapsed_s,armed,found,status" in frames_csv
    assert "test frame" in frames_csv

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["frame_count"] == 1
    assert metadata["outcome"] == "test complete"
    assert metadata["views"] == ["raw"]
