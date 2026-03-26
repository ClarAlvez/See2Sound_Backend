from pathlib import Path

from pipeline.orchestration.process_video import process_video


def test_process_video_returns_complete_structure():
    result = process_video(
        video_path="data/raw_videos/teste.mp4",
        output_base_dir="data/output",
        frame_interval_seconds=1.0,
        scene_threshold=20.0,
        min_scene_gap_seconds=0.5,
    )

    assert "metadata" in result
    assert "audio" in result
    assert "frames" in result
    assert "scenes" in result

def test_process_video_creates_audio_and_frames():
    result = process_video(
        video_path="data/raw_videos/teste.mp4",
        output_base_dir="data/output",
        frame_interval_seconds=1.0,
        scene_threshold=20.0,
        min_scene_gap_seconds=0.5,
    )

    audio_path = Path(result["audio"]["audio_file_path"])
    frames_output_dir = Path(result["frames"]["frames_output_dir"])

    assert audio_path.exists()
    assert frames_output_dir.exists()
    assert result["frames"]["saved_frames_count"] > 0

def test_process_video_detects_scenes():
    result = process_video(
        video_path="data/raw_videos/teste.mp4",
        output_base_dir="data/output",
        frame_interval_seconds=1.0,
        scene_threshold=20.0,
        min_scene_gap_seconds=0.5,
    )

    assert result["scenes"]["scene_change_count"] >= 1
    assert isinstance(result["scenes"]["scene_timestamps_seconds"], list)