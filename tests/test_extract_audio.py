from pathlib import Path
from pipeline.audio.extractor import extract_audio


def test_should_create_audio_file():
    result = extract_audio(
        video_path="data/raw_videos/teste.mp4",
        output_dir="data/extracted_audio"
    )

    audio_path = Path(result["audio_file_path"])

    assert audio_path.exists(), f"Áudio não foi criado em {audio_path}"


def test_should_return_valid_metadata():
    result = extract_audio(
        video_path="data/raw_videos/teste.mp4",
        output_dir="data/extracted_audio"
    )

    assert "audio_file_name" in result
    assert result["audio_format"] == ".wav"