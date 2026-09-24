from pathlib import Path

from fastapi import UploadFile


CHUNK_SIZE = 1024 * 1024


class UploadTooLargeError(Exception):
    pass


async def save_upload_file(
    upload: UploadFile,
    destination: Path,
    max_size_bytes: int,
) -> int:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_size = 0

    try:
        with destination.open("wb") as output_file:
            while True:
                chunk = await upload.read(
                    CHUNK_SIZE
                )

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > max_size_bytes:
                    raise UploadTooLargeError(
                        "O arquivo enviado excede "
                        "o tamanho máximo permitido."
                    )

                output_file.write(chunk)

    except Exception:
        if destination.exists():
            destination.unlink()

        raise

    finally:
        await upload.close()

    return total_size