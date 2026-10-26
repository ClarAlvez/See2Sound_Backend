from pipeline.video.scenes import detect_scene_changes


def test_detect_scene_changes_returns_expected_structure():
    result = detect_scene_changes(
        video_path="data/raw_videos/teste.mp4",
        threshold=20.0,
        min_scene_gap_seconds=0.5
    )

    assert "source_video_name" in result
    assert "scene_change_count" in result
    assert "scene_timestamps_seconds" in result
    assert isinstance(result["scene_timestamps_seconds"], list)


def test_detect_scene_changes_has_at_least_one_scene():
    result = detect_scene_changes(
        video_path="data/raw_videos/teste.mp4",
        threshold=20.0,
        min_scene_gap_seconds=0.5
    )

    assert result["scene_change_count"] >= 1
    assert result["scene_timestamps_seconds"][0] == 0


def test_detect_scene_changes_timestamps_are_sorted():
    result = detect_scene_changes(
        video_path="data/raw_videos/teste.mp4",
        threshold=20.0,
        min_scene_gap_seconds=0.5
    )

    timestamps = result["scene_timestamps_seconds"]
    assert timestamps == sorted(timestamps)
    
