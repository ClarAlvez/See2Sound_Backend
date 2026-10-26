from pathlib import Path

from pipeline.video.frames import extract_frames


def test_extract_frames():
    result = extract_frames(
        video_path="data/raw_videos/corrida.mp4",
        output_dir="data/extracted_frames",
        interval_seconds=1.0
    )

    assert result["saved_frames_count"] > 0
    assert Path(result["frames_output_dir"]).exists()
    assert len(result["frame_paths"]) == result["saved_frames_count"]