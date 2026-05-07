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


def build_json_output_path(
    output_dir: Path,
    file_name: str = "process_video_result.json"
) -> Path:
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
        video_path="data/raw_videos/novo.mp4",
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

    assert "extraction" in result["audio"]
    assert "speech_analysis" in result["audio"]

    audio_extraction = result["audio"]["extraction"]

    possible_audio_keys = [
        "audio_path",
        "audio_file_path",
        "output_audio_path",
    ]

    audio_path = None
    for key in possible_audio_keys:
        if key in audio_extraction:
            audio_path = Path(audio_extraction[key])
            break

    assert audio_path is not None, "Nenhuma chave de caminho de áudio encontrada"
    assert audio_path.exists()

    speech_analysis = result["audio"]["speech_analysis"]

    assert "language" in speech_analysis
    assert "speech_segments" in speech_analysis
    assert "speech_pauses" in speech_analysis
    assert "stats" in speech_analysis

    assert isinstance(speech_analysis["speech_segments"], list)
    assert isinstance(speech_analysis["speech_pauses"], list)

    assert speech_analysis["stats"]["total_segments"] >= 0

    if speech_analysis["speech_segments"]:
        first_segment = speech_analysis["speech_segments"][0]

        assert "start" in first_segment
        assert "end" in first_segment
        assert "text" in first_segment
        assert "duration" in first_segment

    assert Path(result["frames"]["frames_output_dir"]).exists()
    assert result["frames"]["saved_frames_count"] > 0

    assert isinstance(result["scenes"]["scene_timestamps_seconds"], list)
    assert result["scenes"]["scene_change_count"] >= 1

    process_tests_dir = ensure_output_directory("data/process_tests")
    json_output_path = build_json_output_path(process_tests_dir)

    save_result_to_json(result, json_output_path)

    assert json_output_path.exists()