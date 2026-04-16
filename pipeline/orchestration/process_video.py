from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.video.metadata import get_video_metadata
from pipeline.audio.extractor import extract_audio
from pipeline.audio.whisperer import WhisperPauseDetector
from pipeline.video.frames import extract_frames
from pipeline.video.scenes import detect_scene_changes


def validate_video_path(video_path: str) -> Path:
    """
    Valida se o caminho do vídeo existe e é um arquivo.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    if not path.is_file():
        raise ValueError(f"O caminho informado não é um arquivo: {video_path}")

    return path


def ensure_output_base_directory(output_base_dir: str) -> Path:
    """
    Garante que o diretório base de saída exista.
    """
    path = Path(output_base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_directories(output_base_dir: Path) -> Dict[str, Path]:
    """
    Cria e organiza os diretórios usados pela pipeline.
    """
    audio_dir = output_base_dir / "audio"
    frames_dir = output_base_dir / "frames"
    scene_dir = output_base_dir / "scene_data"

    audio_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    return {
        "audio_dir": audio_dir,
        "frames_dir": frames_dir,
        "scene_dir": scene_dir,
    }


def process_metadata(video_path: str) -> Dict[str, Any]:
    """
    Processa os metadados do vídeo.
    """
    return get_video_metadata(video_path)


def process_audio(video_path: str, audio_output_dir: Path) -> Dict[str, Any]:
    """
    Extrai o áudio do vídeo.
    """
    return extract_audio(
        video_path=video_path,
        output_dir=str(audio_output_dir)
    )


def resolve_audio_path(audio_result: Dict[str, Any]) -> Path:
    """
    Resolve o caminho do áudio extraído a partir do retorno do extractor.
    """
    possible_keys = [
        "audio_path",
        "audio_file_path",
        "output_audio_path",
        "path",
        "file_path",
    ]

    for key in possible_keys:
        value = audio_result.get(key)
        if value:
            audio_path = Path(value)
            if audio_path.exists():
                return audio_path

    raise ValueError(
        "Não foi possível localizar o caminho do áudio extraído no retorno de extract_audio."
    )


def analyze_extracted_audio(
    audio_result: Dict[str, Any],
    whisper_model_size: str = "small",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    whisper_language: Optional[str] = None,
    whisper_beam_size: int = 5,
    whisper_vad_filter: bool = True,
    min_pause_duration: float = 0.5,
    refine_with_detected_language: bool = True,
) -> Dict[str, Any]:
    """
    Analisa o áudio extraído com Whisper.
    """
    audio_path = resolve_audio_path(audio_result)

    detector = WhisperPauseDetector(
        model_size=whisper_model_size,
        device=whisper_device,
        compute_type=whisper_compute_type,
    )

    whisper_result = detector.analyze_audio(
        audio_path=audio_path,
        language=whisper_language,
        beam_size=whisper_beam_size,
        vad_filter=whisper_vad_filter,
        min_pause_duration=min_pause_duration,
        refine_with_detected_language=refine_with_detected_language,
    )

    return whisper_result


def process_frames(
    video_path: str,
    frames_output_dir: Path,
    interval_seconds: float
) -> Dict[str, Any]:
    """
    Extrai frames do vídeo.
    """
    return extract_frames(
        video_path=video_path,
        output_dir=str(frames_output_dir),
        interval_seconds=interval_seconds
    )


def process_scenes(
    video_path: str,
    threshold: float,
    min_scene_gap_seconds: float
) -> Dict[str, Any]:
    """
    Detecta mudanças de cena no vídeo.
    """
    return detect_scene_changes(
        video_path=video_path,
        threshold=threshold,
        min_scene_gap_seconds=min_scene_gap_seconds
    )


def build_processing_result(
    video_path: Path,
    metadata_result: Dict[str, Any],
    audio_result: Dict[str, Any],
    whisper_result: Dict[str, Any],
    frames_result: Dict[str, Any],
    scenes_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Monta o resultado final da pipeline.
    """
    return {
        "source_video_name": video_path.name,
        "source_video_path": str(video_path.resolve()),
        "metadata": metadata_result,
        "audio": {
            "extraction": audio_result,
            "speech_analysis": whisper_result,
        },
        "frames": frames_result,
        "scenes": scenes_result,
    }


def process_video(
    video_path: str,
    output_base_dir: str = "data/output",
    frame_interval_seconds: float = 1.0,
    scene_threshold: float = 20.0,
    min_scene_gap_seconds: float = 0.5,
    whisper_model_size: str = "small",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    whisper_language: Optional[str] = None,
    whisper_beam_size: int = 5,
    whisper_vad_filter: bool = True,
    min_pause_duration: float = 0.5,
    refine_with_detected_language: bool = True,
) -> Dict[str, Any]:
    """
    Função principal da pipeline de vídeo.

    Etapas:
    - obtém metadados
    - extrai áudio
    - analisa fala e pausas com Whisper
    - extrai frames
    - detecta mudanças de cena
    """
    validated_video_path = validate_video_path(video_path)
    validated_output_base_dir = ensure_output_base_directory(output_base_dir)
    output_dirs = build_output_directories(validated_output_base_dir)

    metadata_result = process_metadata(str(validated_video_path))

    audio_result = process_audio(
        str(validated_video_path),
        output_dirs["audio_dir"]
    )

    whisper_result = analyze_extracted_audio(
        audio_result=audio_result,
        whisper_model_size=whisper_model_size,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_language=whisper_language,
        whisper_beam_size=whisper_beam_size,
        whisper_vad_filter=whisper_vad_filter,
        min_pause_duration=min_pause_duration,
        refine_with_detected_language=refine_with_detected_language,
    )

    frames_result = process_frames(
        str(validated_video_path),
        output_dirs["frames_dir"],
        frame_interval_seconds
    )

    scenes_result = process_scenes(
        str(validated_video_path),
        scene_threshold,
        min_scene_gap_seconds
    )

    return build_processing_result(
        video_path=validated_video_path,
        metadata_result=metadata_result,
        audio_result=audio_result,
        whisper_result=whisper_result,
        frames_result=frames_result,
        scenes_result=scenes_result,
    )