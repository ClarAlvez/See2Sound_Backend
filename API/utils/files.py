from pathlib import Path
import logging
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

CHUNK_SIZE = 1024 * 1024
logger = logging.getLogger(__name__)

class UploadTooLargeError(Exception):
    pass

async def save_upload_file(upload: UploadFile, destination: Path, max_size_bytes: int) -> int:
    """Copy the spooled upload off the event loop; discard partial copies."""
    def copy_upload():
        destination.parent.mkdir(parents=True, exist_ok=True)
        total_size = 0
        created = False
        try:
            with destination.open('xb') as output_file:
                created = True
                while chunk := upload.file.read(CHUNK_SIZE):
                    total_size += len(chunk)
                    if total_size > max_size_bytes:
                        raise UploadTooLargeError('O arquivo enviado excede o tamanho máximo permitido.')
                    output_file.write(chunk)
        except BaseException:
            if created:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Failed to remove this operation's partial upload")
            raise
        return total_size
    try:
        return await run_in_threadpool(copy_upload)
    finally:
        try:
            await upload.close()
        except OSError:
            logger.exception("Failed to close upload spool")
