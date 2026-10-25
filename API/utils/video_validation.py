"""Bound native decoder hangs to a subprocess before scheduling any AI work."""
import json
import logging
import subprocess
import sys
from pathlib import Path
from API.core.config import settings

logger = logging.getLogger(__name__)

class InvalidVideoError(ValueError):
    pass

class VideoValidationUnavailable(RuntimeError):
    pass


def validate_uploaded_video(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise InvalidVideoError('O vídeo enviado está vazio ou incompleto.')
    try:
        # Preserve the current interpreter's import paths (also supports isolated tests).
        command = [sys.executable, '-c',
                   'import sys,runpy; sys.path[:0]=' + repr(sys.path) +
                   '; runpy.run_module("API.utils.video_probe",run_name="__main__")', str(path)]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                timeout=settings.video_validation_timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise InvalidVideoError('A leitura do vídeo excedeu o tempo limite. Verifique o arquivo e o codec.') from error
    except OSError as error:
        raise VideoValidationUnavailable('Não foi possível iniciar a validação de vídeo.') from error
    try:
        metadata = json.loads(result.stdout)
        if not isinstance(metadata, dict):
            raise ValueError('Invalid probe response')
    except (ValueError, TypeError) as error:
        logger.error('Video probe failed returncode=%s', result.returncode)
        raise VideoValidationUnavailable('O validador de vídeo está indisponível. Verifique OpenCV e o Python.') from error
    if result.returncode == 2 and isinstance(metadata.get('error'), str):
        raise InvalidVideoError(metadata['error'])
    if result.returncode != 0:
        raise VideoValidationUnavailable('Falha no validador de vídeo.')
    if settings.max_video_duration_seconds > 0 and metadata['duration_seconds'] > settings.max_video_duration_seconds:
        raise InvalidVideoError('O vídeo excede a duração máxima configurada.')
    return metadata
