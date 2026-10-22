from pathlib import Path
from typing import Any, Dict
import wave

from moviepy import VideoFileClip


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2


def validate_video_path(
    video_path: str,
) -> Path:
    """
    Valida se o caminho do vídeo existe
    e se realmente aponta para um arquivo.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Vídeo não encontrado: "
            f"{video_path}"
        )

    if not path.is_file():
        raise ValueError(
            "O caminho informado não é "
            f"um arquivo: {video_path}"
        )

    return path


def ensure_output_directory(
    output_dir: str,
) -> Path:
    """
    Garante que o diretório de saída exista.
    """
    path = Path(output_dir)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def build_output_audio_path(
    video_path: Path,
    output_dir: Path,
    extension: str = ".wav",
) -> Path:
    """
    Monta o caminho final do WAV.

    Exemplo:
        video.mp4
        ->
        video.wav
    """
    return (
        output_dir
        / f"{video_path.stem}{extension}"
    )


def load_video_clip(
    video_path: Path,
) -> VideoFileClip:
    """
    Carrega o vídeo com MoviePy.
    """
    try:
        return VideoFileClip(
            str(video_path)
        )

    except Exception as exc:
        raise ValueError(
            "Não foi possível carregar "
            f"o vídeo: {video_path}"
        ) from exc


def get_video_duration(
    video_clip: VideoFileClip,
    video_path: Path,
) -> float:
    """
    Obtém e valida a duração do vídeo.
    """
    duration = getattr(
        video_clip,
        "duration",
        None,
    )

    if duration is None:
        raise ValueError(
            "Não foi possível determinar "
            "a duração do vídeo: "
            f"{video_path}"
        )

    try:
        duration = float(
            duration
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Duração inválida para o vídeo: "
            f"{video_path}"
        ) from exc

    if duration <= 0:
        raise ValueError(
            "O vídeo possui duração inválida: "
            f"{duration}"
        )

    return duration


def has_audio_track(
    video_clip: VideoFileClip,
) -> bool:
    """
    Informa se o vídeo possui uma faixa
    de áudio reconhecida pelo MoviePy.
    """
    return (
        video_clip.audio
        is not None
    )


def write_audio_file(
    video_clip: VideoFileClip,
    output_audio_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """
    Extrai a faixa de áudio original para
    WAV PCM 16-bit.
    """
    if video_clip.audio is None:
        raise ValueError(
            "O vídeo não possui "
            "faixa de áudio."
        )

    video_clip.audio.write_audiofile(
        str(
            output_audio_path
        ),
        codec="pcm_s16le",
        fps=sample_rate,
        logger=None,
    )


def write_silent_audio_file(
    output_audio_path: Path,
    duration_seconds: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> None:
    """
    Cria um WAV silencioso com a mesma
    duração do vídeo.

    O arquivo é criado em pequenos blocos
    para não manter todo o áudio em memória.

    Formato:
    - PCM
    - 16-bit
    - 44100 Hz
    - mono
    """
    if duration_seconds <= 0:
        raise ValueError(
            "A duração para geração de "
            "silêncio deve ser maior "
            "que zero."
        )

    output_audio_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_frames = int(
        round(
            duration_seconds
            * sample_rate
        )
    )

    if total_frames <= 0:
        total_frames = 1

    # Escrevemos no máximo 1 segundo
    # por iteração.
    frames_per_chunk = sample_rate

    bytes_per_frame = (
        channels
        * sample_width
    )

    with wave.open(
        str(output_audio_path),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            channels
        )

        wav_file.setsampwidth(
            sample_width
        )

        wav_file.setframerate(
            sample_rate
        )

        remaining_frames = (
            total_frames
        )

        while remaining_frames > 0:
            chunk_frames = min(
                frames_per_chunk,
                remaining_frames,
            )

            silence = (
                b"\x00"
                * (
                    chunk_frames
                    * bytes_per_frame
                )
            )

            wav_file.writeframes(
                silence
            )

            remaining_frames -= (
                chunk_frames
            )


def build_extraction_result(
    video_path: Path,
    output_audio_path: Path,
    duration_seconds: float,
    has_original_audio: bool,
) -> Dict[str, Any]:
    """
    Retorna os dados da etapa de áudio.

    `generated_silence` permite identificar
    claramente quando o áudio foi criado
    artificialmente porque o vídeo não
    possuía faixa sonora.
    """
    generated_silence = (
        not has_original_audio
    )

    return {
        "source_video_name": (
            video_path.name
        ),

        "source_video_path": str(
            video_path.resolve()
        ),

        "audio_file_name": (
            output_audio_path.name
        ),

        "audio_file_path": str(
            output_audio_path.resolve()
        ),

        "audio_format": (
            output_audio_path.suffix
        ),

        "duration_seconds": round(
            duration_seconds,
            4,
        ),

        "sample_rate": (
            DEFAULT_SAMPLE_RATE
        ),

        "has_original_audio": (
            has_original_audio
        ),

        "generated_silence": (
            generated_silence
        ),

        "audio_source": (
            "original"
            if has_original_audio
            else "generated_silence"
        ),
    }


def extract_audio(
    video_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Prepara a faixa de áudio utilizada pelo
    restante da pipeline.

    Caso 1:
        vídeo possui áudio
        -> extrai normalmente.

    Caso 2:
        vídeo não possui áudio
        -> cria WAV silencioso com a
           mesma duração do vídeo.

    Em ambos os casos o restante da pipeline
    recebe sempre um arquivo WAV válido.
    """
    validated_video_path = (
        validate_video_path(
            video_path
        )
    )

    validated_output_dir = (
        ensure_output_directory(
            output_dir
        )
    )

    output_audio_path = (
        build_output_audio_path(
            video_path=(
                validated_video_path
            ),
            output_dir=(
                validated_output_dir
            ),
        )
    )

    video_clip = load_video_clip(
        validated_video_path
    )

    try:
        duration_seconds = (
            get_video_duration(
                video_clip=(
                    video_clip
                ),
                video_path=(
                    validated_video_path
                ),
            )
        )

        original_audio_exists = (
            has_audio_track(
                video_clip
            )
        )

        if original_audio_exists:
            print(
                "[Audio] Faixa de áudio "
                "original encontrada."
            )

            write_audio_file(
                video_clip=(
                    video_clip
                ),
                output_audio_path=(
                    output_audio_path
                ),
            )

            print(
                "[Audio] Áudio original "
                "extraído com sucesso."
            )

        else:
            print(
                "[Audio] Vídeo sem faixa "
                "de áudio."
            )

            print(
                "[Audio] Criando WAV "
                "silencioso com duração "
                f"de {duration_seconds:.3f}s."
            )

            write_silent_audio_file(
                output_audio_path=(
                    output_audio_path
                ),
                duration_seconds=(
                    duration_seconds
                ),
            )

            print(
                "[Audio] WAV silencioso "
                "criado com sucesso."
            )

        return (
            build_extraction_result(
                video_path=(
                    validated_video_path
                ),
                output_audio_path=(
                    output_audio_path
                ),
                duration_seconds=(
                    duration_seconds
                ),
                has_original_audio=(
                    original_audio_exists
                ),
            )
        )

    finally:
        video_clip.close()