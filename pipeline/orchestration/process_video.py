from pathlib import Path
from typing import Any, Dict

from pipeline.video.metadata import get_video_metadata
from pipeline.audio.extractor import extract_audio
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
        "audio": audio_result,
        "frames": frames_result,
        "scenes": scenes_result,
    }


def process_video(
    video_path: str,
    output_base_dir: str = "data/output",
    frame_interval_seconds: float = 1.0,
    scene_threshold: float = 20.0,
    min_scene_gap_seconds: float = 0.5,
) -> Dict[str, Any]:
    """
    Função principal da pipeline de vídeo.

    Etapas:
    - obtém metadados
    - extrai áudio
    - extrai frames
    - detecta mudanças de cena
    """
    validated_video_path = validate_video_path(video_path)
    validated_output_base_dir = ensure_output_base_directory(output_base_dir)
    output_dirs = build_output_directories(validated_output_base_dir)

    metadata_result = process_metadata(str(validated_video_path))
    audio_result = process_audio(str(validated_video_path), output_dirs["audio_dir"])
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
        frames_result=frames_result,
        scenes_result=scenes_result,
    )