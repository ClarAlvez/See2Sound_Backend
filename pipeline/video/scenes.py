from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


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


def convert_frame_to_gray(frame: np.ndarray) -> np.ndarray:
    """
    Converte um frame para escala de cinza.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def resize_frame(frame: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
    """
    Redimensiona o frame para acelerar comparações.
    """
    return cv2.resize(frame, (width, height))


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Pré-processa o frame para comparação:
    - redimensiona
    - converte para escala de cinza
    """
    resized = resize_frame(frame)
    gray = convert_frame_to_gray(resized)
    return gray


def calculate_frame_difference(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Calcula a diferença média absoluta entre dois frames.
    """
    diff = cv2.absdiff(frame_a, frame_b)
    return float(np.mean(diff))


def frame_index_to_timestamp_seconds(frame_index: int, fps: float) -> float:
    """
    Converte índice de frame em timestamp (segundos).
    """
    return frame_index / fps


def detect_scene_changes_from_capture(
    capture: cv2.VideoCapture,
    fps: float,
    threshold: float = 20.0,
    min_scene_gap_seconds: float = 0.5
) -> Tuple[int, List[float], List[float]]:
    """
    Detecta mudanças de cena comparando frames consecutivos.
    Retorna:
    - quantidade de mudanças detectadas
    - timestamps das mudanças
    - diferenças medidas
    """
    previous_frame: Optional[np.ndarray] = None
    current_frame_index = 0

    scene_timestamps: List[float] = [0.0]  # considera início do vídeo como primeira cena
    frame_differences: List[float] = []

    min_scene_gap_frames = max(int(fps * min_scene_gap_seconds), 1)
    last_detected_scene_frame = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        processed_frame = preprocess_frame(frame)

        if previous_frame is not None:
            difference = calculate_frame_difference(previous_frame, processed_frame)
            frame_differences.append(difference)

            enough_gap = (current_frame_index - last_detected_scene_frame) >= min_scene_gap_frames

            if difference >= threshold and enough_gap:
                timestamp = frame_index_to_timestamp_seconds(current_frame_index, fps)
                scene_timestamps.append(round(timestamp, 3))
                last_detected_scene_frame = current_frame_index

        previous_frame = processed_frame
        current_frame_index += 1

    scene_change_count = len(scene_timestamps)
    return scene_change_count, scene_timestamps, frame_differences


def build_scene_detection_result(
    video_path: Path,
    threshold: float,
    min_scene_gap_seconds: float,
    scene_change_count: int,
    scene_timestamps: List[float]
) -> Dict[str, Any]:
    """
    Monta o dicionário final com o resultado da detecção de cenas.
    """
    return {
        "source_video_name": video_path.name,
        "source_video_path": str(video_path.resolve()),
        "threshold": threshold,
        "min_scene_gap_seconds": min_scene_gap_seconds,
        "scene_change_count": scene_change_count,
        "scene_timestamps_seconds": scene_timestamps,
    }


def detect_scene_changes(
    video_path: str,
    threshold: float = 20.0,
    min_scene_gap_seconds: float = 0.5
) -> Dict[str, Any]:
    """
    Função principal para detectar mudanças de cena em um vídeo.
    """
    validated_video_path = validate_video_path(video_path)
    capture = open_video_capture(validated_video_path)

    try:
        fps = get_video_fps(capture)
        scene_change_count, scene_timestamps, _ = detect_scene_changes_from_capture(
            capture=capture,
            fps=fps,
            threshold=threshold,
            min_scene_gap_seconds=min_scene_gap_seconds
        )

        return build_scene_detection_result(
            video_path=validated_video_path,
            threshold=threshold,
            min_scene_gap_seconds=min_scene_gap_seconds,
            scene_change_count=scene_change_count,
            scene_timestamps=scene_timestamps
        )

    finally:
        capture.release()