from pathlib import Path
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


def get_video_fps(capture: cv2.VideoCapture) -> float:
    """
    Retorna o FPS do vídeo.
    """
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        raise ValueError("FPS inválido ou não encontrado no vídeo.")

    return fps


def release_video_capture(capture: cv2.VideoCapture) -> None:
    """
    Libera o recurso do vídeo.
    """
    capture.release()