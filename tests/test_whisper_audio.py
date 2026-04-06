from pathlib import Path

from pipeline.audio.whisperer import WhisperPauseDetector
from pipeline.audio.utils import save_json


def test_whisper_audio_analysis():
    """
    Teste simples:
    - pega um áudio já extraído
    - roda whisper
    - gera json com segmentos e pausas
    """

    audio_file = Path("data/output/audio/teste.wav")
    output_file = Path("data/process_tests/whisper_audio_result.json")

    detector = WhisperPauseDetector(
        model_size="small",
        device="cpu",
        compute_type="int8"
    )

    result = detector.analyze_audio(
        audio_path=audio_file,
        language="pt",
        beam_size=5,
        vad_filter=True,
        min_pause_duration=0.5
    )

    save_json(result, output_file)

    assert output_file.exists()
    assert "speech_segments" in result
    assert "speech_pauses" in result