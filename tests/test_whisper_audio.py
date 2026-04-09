from __future__ import annotations

import json
from pathlib import Path

from pipeline.audio.whisperer import WhisperPauseDetector


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def test_whisper_audio_analysis():
    """
    Teste de integração simples do whisperer.

    Fluxo:
    - lê um arquivo de áudio
    - analisa idioma, segmentos e pausas
    - salva o resultado em JSON
    """
    audio_file = Path("data/output/audio/teste2.wav")
    output_file = Path("data/process_tests/whisper_audio_result.json")

    assert audio_file.exists(), f"Arquivo de áudio não encontrado: {audio_file}"

    detector = WhisperPauseDetector(
        model_size="small",
        device="cpu",
        compute_type="int8",
    )

    result = detector.analyze_audio(
        audio_path=audio_file,
        language=None,  # detecção automática
        beam_size=5,
        vad_filter=True,
        min_pause_duration=0.5,
        refine_with_detected_language=True,
    )

    save_json(result, output_file)

    assert output_file.exists()
    assert "language" in result
    assert "speech_segments" in result
    assert "speech_pauses" in result
    assert "stats" in result
    assert isinstance(result["speech_segments"], list)
    assert isinstance(result["speech_pauses"], list)