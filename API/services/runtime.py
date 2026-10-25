"""Local bounded admission and one model pipeline at a time."""
import logging
import os
import threading
from API.core.config import settings
from API.core.errors import StoredDataError
from API.services.job_store import read_job, update_job, utc_now

logger = logging.getLogger(__name__)
_pipeline_lock = threading.Lock()
_slots = threading.BoundedSemaphore(settings.max_pending_generations)


def reserve_generation_slot():
    return _slots.acquire(blocking=False)


def release_generation_slot():
    _slots.release()


def run_admitted_generation(job_id, video_path, options, worker):
    try:
        with _pipeline_lock:
            worker(job_id, video_path, options)
    finally:
        release_generation_slot()


class RuntimeLease:
    """An OS lock prevents two API processes from sharing the same job storage."""
    def __enter__(self):
        self.file = (settings.jobs_dir / '_runtime.lock').open('a+b')
        try:
            self.file.seek(0, os.SEEK_END)
            if self.file.tell() == 0:
                self.file.write(b'0')
                self.file.flush()
            self.file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            raise RuntimeError('Outra API está usando os mesmos jobs. Execute com um único worker.') from error
        return self

    def __exit__(self, *args):
        # Closing the descriptor releases the OS lock, including after a crash.
        self.file.close()


def recover_interrupted_jobs():
    """Called only while holding the exclusive runtime lease, before serving traffic."""
    for path in settings.jobs_dir.glob('*.json'):
        try:
            job = read_job(path.stem)
            if job['status'] in ('queued', 'processing'):
                update_job(path.stem, status='failed', finished_at=utc_now(),
                           stage='interrupted',
                           error='Processamento interrompido por reinicialização da API. Envie o vídeo novamente.')
                logger.warning('job=%s stage=interrupted previous_status=%s', path.stem, job['status'])
        except (OSError, StoredDataError):
            logger.exception('Cannot recover job=%s', path.stem)
