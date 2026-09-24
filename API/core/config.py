import os
from pathlib import Path


class Settings:
    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[2]

        self.runtime_root = (
            self.repo_root
            / "data"
            / "output"
            / "api"
        )

        self.upload_dir = (
            self.runtime_root
            / "_uploads"
        )

        self.jobs_dir = (
            self.runtime_root
            / "_jobs"
        )

        self.generations_dir = (
            self.runtime_root
            / "generations"
        )

        self.max_upload_size_mb = int(
            os.getenv(
                "SEE2SOUND_MAX_UPLOAD_MB",
                "500",
            )
        )

        self.allowed_video_extensions = {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
            ".m4v",
        }

        cors_value = os.getenv(
            "SEE2SOUND_CORS_ORIGINS",
            "*",
        )

        self.cors_origins = [
            origin.strip()
            for origin in cors_value.split(",")
            if origin.strip()
        ]

    @property
    def max_upload_size_bytes(self) -> int:
        return (
            self.max_upload_size_mb
            * 1024
            * 1024
        )

    def create_runtime_directories(self):
        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.jobs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.generations_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


settings = Settings()

settings.create_runtime_directories()