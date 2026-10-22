import os
from pathlib import Path
from typing import Dict


class Settings:
    def __init__(self):
        # ====================================================
        # Raiz do projeto
        # ====================================================

        self.repo_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        # ====================================================
        # Modelos
        # ====================================================

        self.models_dir = (
            self.repo_root
            / "data"
            / "models"
        )

        self.scene_model_path = (
            self._resolve_model_path(
                env_name=(
                    "SEE2SOUND_SCENE_MODEL"
                ),
                default_relative_path=(
                    "Scene/"
                    "scene_net_best.pt"
                ),
            )
        )

        self.person_model_path = (
            self._resolve_model_path(
                env_name=(
                    "SEE2SOUND_PERSON_MODEL"
                ),
                default_relative_path=(
                    "Person/"
                    "person_net_best.pt"
                ),
            )
        )

        self.object_model_path = (
            self._resolve_model_path(
                env_name=(
                    "SEE2SOUND_OBJECT_MODEL"
                ),
                default_relative_path=(
                    "Object/"
                    "object_net_best.pt"
                ),
            )
        )

        self.atmosphere_model_path = (
            self._resolve_model_path(
                env_name=(
                    "SEE2SOUND_ATMOSPHERE_MODEL"
                ),
                default_relative_path=(
                    "Atmosphere/"
                    "atmosphere_net_best.pt"
                ),
            )
        )

        self.action_model_path = (
            self._resolve_model_path(
                env_name=(
                    "SEE2SOUND_ACTION_MODEL"
                ),
                default_relative_path=(
                    "Actions/"
                    "action_net_best.pt"
                ),
            )
        )

        self.narrative_model_path = (
            self._resolve_model_path(
                env_name=(
                    "SEE2SOUND_NARRATIVE_MODEL"
                ),
                default_relative_path=(
                    "qwen/"
                    "qwen2.5-3b-instruct-"
                    "q4_k_m.gguf"
                ),
            )
        )

        # ====================================================
        # Runtime da API
        # ====================================================

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

        # ====================================================
        # Uploads
        # ====================================================

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

        # ====================================================
        # CORS
        # ====================================================

        cors_value = os.getenv(
            "SEE2SOUND_CORS_ORIGINS",
            "*",
        )

        self.cors_origins = [
            origin.strip()
            for origin
            in cors_value.split(",")
            if origin.strip()
        ]

    # ========================================================
    # Caminhos
    # ========================================================

    def _resolve_model_path(
        self,
        env_name: str,
        default_relative_path: str,
    ) -> Path:
        """
        Resolve o caminho de um modelo.

        É possível sobrescrever qualquer
        modelo por variável de ambiente.

        Caminhos relativos são interpretados
        em relação à raiz do repositório.
        """

        environment_value = (
            os.getenv(
                env_name
            )
        )

        if environment_value:
            path = Path(
                environment_value
            ).expanduser()

            if not path.is_absolute():
                path = (
                    self.repo_root
                    / path
                )

            return path.resolve()

        return (
            self.models_dir
            / default_relative_path
        ).resolve()

    # ========================================================
    # Validação dos modelos
    # ========================================================

    def get_model_paths(
        self,
    ) -> Dict[str, Path]:
        return {
            "scene": (
                self.scene_model_path
            ),
            "person": (
                self.person_model_path
            ),
            "object": (
                self.object_model_path
            ),
            "atmosphere": (
                self.atmosphere_model_path
            ),
            "action": (
                self.action_model_path
            ),
            "narrative": (
                self.narrative_model_path
            ),
        }

    def get_missing_models(
        self,
        run_spectra: bool = True,
        run_narrative: bool = True,
    ) -> Dict[str, Path]:
        """
        Retorna apenas os modelos necessários
        para a execução solicitada que não
        foram encontrados.
        """

        required: Dict[
            str,
            Path
        ] = {}

        if run_spectra:
            required.update(
                {
                    "scene": (
                        self.scene_model_path
                    ),
                    "person": (
                        self.person_model_path
                    ),
                    "object": (
                        self.object_model_path
                    ),
                    "atmosphere": (
                        self.atmosphere_model_path
                    ),
                    "action": (
                        self.action_model_path
                    ),
                }
            )

        if (
            run_spectra
            and run_narrative
        ):
            required[
                "narrative"
            ] = (
                self.narrative_model_path
            )

        return {
            name: path
            for name, path
            in required.items()
            if not path.is_file()
        }

    def validate_model_files(
        self,
        run_spectra: bool = True,
        run_narrative: bool = True,
    ) -> None:
        """
        Falha explicitamente caso um modelo
        necessário não exista.

        Isso evita a situação anterior em que
        Person/Object/Atmosphere/Actions eram
        silenciosamente ignorados.
        """

        missing = (
            self.get_missing_models(
                run_spectra=(
                    run_spectra
                ),
                run_narrative=(
                    run_narrative
                ),
            )
        )

        if not missing:
            return

        lines = [
            "Modelos necessários "
            "não encontrados:"
        ]

        for (
            model_name,
            model_path,
        ) in missing.items():
            lines.append(
                f"- {model_name}: "
                f"{model_path}"
            )

        raise FileNotFoundError(
            "\n".join(
                lines
            )
        )

    # ========================================================
    # Upload
    # ========================================================

    @property
    def max_upload_size_bytes(
        self,
    ) -> int:
        return (
            self.max_upload_size_mb
            * 1024
            * 1024
        )

    # ========================================================
    # Diretórios runtime
    # ========================================================

    def create_runtime_directories(
        self,
    ):
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