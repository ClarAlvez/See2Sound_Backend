import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4
from API.core.config import settings
from API.core.errors import StoredDataError

_store_lock = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_job_path(job_id: str) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', job_id):
        raise FileNotFoundError('Geração não encontrada.')
    return settings.jobs_dir / f'{job_id}.json'


def create_job(job_id: str, original_filename: str, input_path: str,
               upload_size: int, options: Dict[str, Any]) -> Dict[str, Any]:
    job = {'job_id': job_id, 'status': 'queued', 'original_filename': original_filename,
           'input_path': input_path, 'upload_size': upload_size, 'options': options,
           'created_at': utc_now(), 'started_at': None, 'finished_at': None,
           'result_path': None, 'error': None}
    with _store_lock:
        if get_job_path(job_id).exists():
            raise FileExistsError('ID de geração já existe.')
        _write_job(job)
    return job


def read_job(job_id: str) -> Dict[str, Any]:
    path = get_job_path(job_id)
    with _store_lock:
        if not path.is_file():
            raise FileNotFoundError(f'Job não encontrado: {job_id}')
        try:
            with path.open('r', encoding='utf-8') as file:
                job = json.load(file)
            required = {'job_id', 'status', 'original_filename', 'created_at'}
            if not isinstance(job, dict) or not required.issubset(job) or job['job_id'] != job_id:
                raise ValueError('Invalid job record')
            if not all(isinstance(job[key], str) for key in required):
                raise ValueError('Invalid job field types')
            if not isinstance(job.get('options', {}), dict):
                raise ValueError('Invalid job options')
            if job['status'] not in {'queued', 'processing', 'completed', 'failed'}:
                raise ValueError('Invalid job state')
            return job
        except (OSError, ValueError, TypeError) as error:
            raise StoredDataError('Registro da geração está ilegível. Consulte os logs do servidor.') from error


def update_job(job_id: str, **changes) -> Dict[str, Any]:
    with _store_lock:
        job = read_job(job_id)
        job.update(changes)
        _write_job(job)
        return job


def _write_job(job: Dict[str, Any]):
    path = get_job_path(job['job_id'])
    temporary_path = path.with_name(f'{path.stem}.{uuid4().hex}.tmp')
    try:
        with temporary_path.open('x', encoding='utf-8') as file:
            json.dump(job, file, indent=4, ensure_ascii=False, default=str)
            file.flush()
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
