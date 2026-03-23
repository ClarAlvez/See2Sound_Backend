from pathlib import Path
from typing import Any, Dict

from moviepy import VideoFileClip


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


def ensure_output_directory(output_dir: str) -> Path:
    """
    Garante que o diretório de saída exista.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_audio_path(video_path: Path, output_dir: Path, extension: str = ".wav") -> Path:
    """
    Monta o caminho final do arquivo de áudio a partir do nome do vídeo.
    """
    return output_dir / f"{video_path.stem}{extension}"


def load_video_clip(video_path: Path) -> VideoFileClip:
    """
    Carrega o vídeo usando MoviePy.
    """
    try:
        return VideoFileClip(str(video_path))
    except Exception as exc:
        raise ValueError(f"Não foi possível carregar o vídeo: {video_path}") from exc


def validate_audio_track(video_clip: VideoFileClip, video_path: Path) -> None:
    """
    Verifica se o vídeo possui faixa de áudio.
    """
    if video_clip.audio is None:
        raise ValueError(f"O vídeo não possui faixa de áudio: {video_path}")


def write_audio_file(video_clip: VideoFileClip, output_audio_path: Path) -> None:
    """
    Escreve o áudio extraído em arquivo.
    """
    video_clip.audio.write_audiofile(
        str(output_audio_path),
        codec="pcm_s16le",
        logger=None
    )


def build_extraction_result(video_path: Path, output_audio_path: Path) -> Dict[str, Any]:
    """
    Retorna um dicionário com informações básicas da extração.
    """
    return {
        "source_video_name": video_path.name,
        "source_video_path": str(video_path.resolve()),
        "audio_file_name": output_audio_path.name,
        "audio_file_path": str(output_audio_path.resolve()),
        "audio_format": output_audio_path.suffix,
    }


def extract_audio(video_path: str, output_dir: str):
    validated_video_path = validate_video_path(video_path)

    validated_output_dir = ensure_output_directory(output_dir)

    output_audio_path = build_output_audio_path(validated_video_path, validated_output_dir)

    video_clip = load_video_clip(validated_video_path)

    try:
        validate_audio_track(video_clip, validated_video_path)
        print("Faixa de áudio encontrada")
        write_audio_file(video_clip, output_audio_path)
        print("Áudio extraído com sucesso")
        return build_extraction_result(validated_video_path, output_audio_path)
    finally:
        video_clip.close()