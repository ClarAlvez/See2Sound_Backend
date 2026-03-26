from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def ensure_output_directory(output_dir: str) -> Path:
    """
    Garante que o diretório de saída exista.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
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


def calculate_frame_interval(fps: float, interval_seconds: float) -> int:
    """
    Calcula a cada quantos frames deve ocorrer a extração.
    Exemplo: fps=30 e interval_seconds=1 => salva 1 frame a cada 30 frames.
    """
    if interval_seconds <= 0:
        raise ValueError("O intervalo em segundos deve ser maior que zero.")

    interval = int(fps * interval_seconds)
    return max(interval, 1)


def build_frame_output_path(
    output_dir: Path,
    video_stem: str,
    frame_index: int
) -> Path:
    """
    Monta o caminho do arquivo de saída para um frame.
    """
    return output_dir / f"{video_stem}_frame_{frame_index:06d}.jpg"


def save_frame(frame, output_path: Path) -> None:
    """
    Salva um frame em disco.
    """
    success = cv2.imwrite(str(output_path), frame)

    if not success:
        raise IOError(f"Não foi possível salvar o frame em: {output_path}")


def extract_frames_from_capture(
    capture: cv2.VideoCapture,
    output_dir: Path,
    video_stem: str,
    frame_interval: int
) -> Tuple[int, List[str]]:
    """
    Percorre o vídeo e salva frames com base no intervalo calculado.
    """
    current_frame_index = 0
    saved_frames_count = 0
    saved_frame_paths: List[str] = []

    while True:
        success, frame = capture.read()

        if not success:
            break

        if current_frame_index % frame_interval == 0:
            output_path = build_frame_output_path(
                output_dir=output_dir,
                video_stem=video_stem,
                frame_index=current_frame_index
            )

            save_frame(frame, output_path)
            saved_frames_count += 1
            saved_frame_paths.append(str(output_path.resolve()))

        current_frame_index += 1

    return saved_frames_count, saved_frame_paths


def build_extraction_result(
    video_path: Path,
    output_dir: Path,
    saved_frames_count: int,
    saved_frame_paths: List[str],
    interval_seconds: float
) -> Dict[str, Any]:
    """
    Retorna um dicionário com informações básicas da extração de frames.
    """
    return {
        "source_video_name": video_path.name,
        "source_video_path": str(video_path.resolve()),
        "frames_output_dir": str(output_dir.resolve()),
        "saved_frames_count": saved_frames_count,
        "interval_seconds": interval_seconds,
        "frame_paths": saved_frame_paths,
    }


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 1.0
) -> Dict[str, Any]:
    """
    Função principal para extrair frames de um vídeo em intervalos fixos.
    """
    validated_video_path = validate_video_path(video_path)
    validated_output_dir = ensure_output_directory(output_dir)

    capture = open_video_capture(validated_video_path)

    try:
        fps = get_video_fps(capture)
        frame_interval = calculate_frame_interval(fps, interval_seconds)

        saved_frames_count, saved_frame_paths = extract_frames_from_capture(
            capture=capture,
            output_dir=validated_output_dir,
            video_stem=validated_video_path.stem,
            frame_interval=frame_interval
        )

        return build_extraction_result(
            video_path=validated_video_path,
            output_dir=validated_output_dir,
            saved_frames_count=saved_frames_count,
            saved_frame_paths=saved_frame_paths,
            interval_seconds=interval_seconds
        )

    finally:
        capture.release()