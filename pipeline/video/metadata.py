from pathlib import Path
from typing import Any, Dict

import cv2

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


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    """
    Abre o vídeo com OpenCV e garante que ele foi carregado corretamente.
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    return capture


def get_fps(capture: cv2.VideoCapture) -> float:
    """
    Retorna o FPS do vídeo.
    """
    return float(capture.get(cv2.CAP_PROP_FPS))


def get_total_frames(capture: cv2.VideoCapture) -> int:
    """
    Retorna o número total de frames do vídeo.
    """
    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


def get_frame_dimensions(capture: cv2.VideoCapture) -> tuple[int, int]:
    """
    Retorna largura e altura do vídeo.
    """
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height


def calculate_duration_seconds(total_frames: int, fps: float) -> float:
    """
    Calcula a duração do vídeo em segundos.
    """
    if fps <= 0:
        return 0.0

    return total_frames / fps


def build_metadata_dict(
    video_path: Path,
    fps: float,
    total_frames: int,
    width: int,
    height: int,
    duration_seconds: float,
) -> Dict[str, Any]:
    """
    Monta o dicionário final de metadados do vídeo.
    """
    return {
        "file_name": video_path.name,
        "file_path": str(video_path.resolve()),
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "duration_seconds": duration_seconds,
    }


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Função principal para obter os metadados básicos de um vídeo.
    """
    path = validate_video_path(video_path)
    capture = open_video_capture(path)

    try:
        fps = get_fps(capture)
        total_frames = get_total_frames(capture)
        width, height = get_frame_dimensions(capture)
        duration_seconds = calculate_duration_seconds(total_frames, fps)

        metadata = build_metadata_dict(
            video_path=path,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
        )

        return metadata

    finally:
        capture.release()