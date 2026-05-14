import argparse
import re
from pathlib import Path

import cv2
import yt_dlp


def sanitize_name(name: str) -> str:
    """
    Transforma o título do vídeo em um nome seguro para pasta/arquivo.
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def download_youtube_video(url: str, output_dir: Path) -> Path:
    """
    Baixa um vídeo do YouTube usando yt-dlp.
    Retorna o caminho do arquivo baixado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "mp4/best",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        title = info.get("title", "youtube_video")
        ext = info.get("ext", "mp4")

        downloaded_path = output_dir / f"{title}.{ext}"

        if downloaded_path.exists():
            return downloaded_path

        # Fallback: procura o arquivo baixado na pasta
        video_files = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.webm")) + list(output_dir.glob("*.mkv"))

        if not video_files:
            raise FileNotFoundError("Nenhum arquivo de vídeo foi encontrado após o download.")

        return video_files[0]


def extract_spaced_frames(
    video_path: Path,
    frames_output_dir: Path,
    interval_seconds: float = 2.0,
    max_frames: int = 100,
) -> int:
    """
    Extrai frames espaçados de um vídeo.
    Exemplo: 1 frame a cada 2 segundos.
    """
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("FPS inválido ou não detectado.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps

    frame_step = int(fps * interval_seconds)

    if frame_step <= 0:
        frame_step = 1

    saved_count = 0
    current_frame = 0

    while current_frame < total_frames and saved_count < max_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        success, frame = capture.read()

        if not success:
            break

        timestamp_seconds = current_frame / fps

        frame_name = f"frame_{saved_count:06d}_t{timestamp_seconds:.2f}.jpg"
        frame_path = frames_output_dir / frame_name

        cv2.imwrite(str(frame_path), frame)

        saved_count += 1
        current_frame += frame_step

    capture.release()

    print(f"Vídeo: {video_path}")
    print(f"Duração aproximada: {duration_seconds:.2f}s")
    print(f"Frames salvos: {saved_count}")
    print(f"Pasta: {frames_output_dir}")

    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Baixa um vídeo do YouTube e extrai frames espaçados."
    )

    parser.add_argument(
        "url",
        help="Link do vídeo do YouTube."
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Nome da pasta de saída. Se não passar, usa youtube_video."
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Intervalo em segundos entre os frames extraídos."
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Quantidade máxima de frames para salvar."
    )

    parser.add_argument(
        "--output-dir",
        default="data/dataset_sources",
        help="Pasta base onde o vídeo e os frames serão salvos."
    )

    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)

    source_name = sanitize_name(args.name or "youtube_video")

    video_output_dir = base_output_dir / source_name / "video"
    frames_output_dir = base_output_dir / source_name / "frames"

    video_path = download_youtube_video(
        url=args.url,
        output_dir=video_output_dir
    )

    extract_spaced_frames(
        video_path=video_path,
        frames_output_dir=frames_output_dir,
        interval_seconds=args.interval,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()

# rodar: python -m tools.extract_youtube_frames "https://www.youtube.com/watch?v=LINK_DO_VIDEO" --name corrida --interval 2 --max-frames 80