import json
from pathlib import Path

from pipeline.orchestration.process_video import process_video


def ensure_output_directory(output_dir: str) -> Path:
    """
    Garante que o diretório de saída exista.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_json_output_path(output_dir: Path, file_name: str = "process_video_result.json") -> Path:
    """
    Monta o caminho do arquivo JSON de saída.
    """
    return output_dir / file_name


def save_result_to_json(result: dict, output_path: Path) -> None:
    """
    Salva o resultado da pipeline em JSON.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


def test_process_video_pipeline():
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

    assert result["metadata"]["total_frames"] > 0
    assert result["metadata"]["fps"] > 0

    assert Path(result["audio"]["audio_file_path"]).exists()
    assert Path(result["frames"]["frames_output_dir"]).exists()
    assert result["frames"]["saved_frames_count"] > 0

    assert isinstance(result["scenes"]["scene_timestamps_seconds"], list)
    assert result["scenes"]["scene_change_count"] >= 1

    process_tests_dir = ensure_output_directory("data/process_tests")
    json_output_path = build_json_output_path(process_tests_dir)

    save_result_to_json(result, json_output_path)

    assert json_output_path.exists()