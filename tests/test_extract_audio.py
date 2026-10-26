
import pytest

pytestmark = pytest.mark.media
from pathlib import Path


def test_should_create_audio_file(tmp_path):
    from pipeline.audio.extractor import extract_audio

    result = extract_audio(
        video_path="data/raw_videos/teste2.mp4",
        output_dir=str(tmp_path / "audio")
    )

    audio_path = Path(result["audio_file_path"])

    assert audio_path.exists(), f"Áudio não foi criado em {audio_path}"


def test_should_return_valid_metadata(tmp_path):
    from pipeline.audio.extractor import extract_audio

    result = extract_audio(
        video_path="data/raw_videos/teste.mp4",
        output_dir=str(tmp_path / "audio")
    )

    assert "audio_file_name" in result
    assert result["audio_format"] == ".wav"