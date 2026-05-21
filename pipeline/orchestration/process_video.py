import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.video.metadata import get_video_metadata
from pipeline.audio.extractor import extract_audio
from pipeline.audio.whisperer import WhisperPauseDetector
from pipeline.video.frames import extract_frames
from pipeline.video.scenes import detect_scene_changes

from ai.spectra.spectra import Spectra


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


def ensure_output_base_directory(output_base_dir: str) -> Path:
    """
    Garante que o diretório base de saída exista.
    """
    path = Path(output_base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_directories(output_base_dir: Path) -> Dict[str, Path]:
    """
    Cria e organiza os diretórios usados pela pipeline.
    """
    audio_dir = output_base_dir / "audio"
    frames_dir = output_base_dir / "frames"
    scene_dir = output_base_dir / "scene_data"
    visual_dir = output_base_dir / "visual_analysis"

    audio_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    return {
        "audio_dir": audio_dir,
        "frames_dir": frames_dir,
        "scene_dir": scene_dir,
        "visual_dir": visual_dir,
    }


def process_metadata(video_path: str) -> Dict[str, Any]:
    """
    Processa os metadados do vídeo.
    """
    return get_video_metadata(video_path)


def process_audio(video_path: str, audio_output_dir: Path) -> Dict[str, Any]:
    """
    Extrai o áudio do vídeo.
    """
    return extract_audio(
        video_path=video_path,
        output_dir=str(audio_output_dir)
    )


def resolve_audio_path(audio_result: Dict[str, Any]) -> Path:
    """
    Resolve o caminho do áudio extraído a partir do retorno do extractor.
    """
    possible_keys = [
        "audio_path",
        "audio_file_path",
        "output_audio_path",
        "path",
        "file_path",
    ]

    for key in possible_keys:
        value = audio_result.get(key)

        if value:
            audio_path = Path(value)

            if audio_path.exists():
                return audio_path

    raise ValueError(
        "Não foi possível localizar o caminho do áudio extraído no retorno de extract_audio."
    )


def analyze_extracted_audio(
    audio_result: Dict[str, Any],
    whisper_model_size: str = "small",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    whisper_language: Optional[str] = None,
    whisper_beam_size: int = 5,
    whisper_vad_filter: bool = True,
    min_pause_duration: float = 0.5,
    refine_with_detected_language: bool = True,
) -> Dict[str, Any]:
    """
    Analisa o áudio extraído com Whisper.
    """
    audio_path = resolve_audio_path(audio_result)

    detector = WhisperPauseDetector(
        model_size=whisper_model_size,
        device=whisper_device,
        compute_type=whisper_compute_type,
    )

    whisper_result = detector.analyze_audio(
        audio_path=audio_path,
        language=whisper_language,
        beam_size=whisper_beam_size,
        vad_filter=whisper_vad_filter,
        min_pause_duration=min_pause_duration,
        refine_with_detected_language=refine_with_detected_language,
    )

    return whisper_result


def process_frames(
    video_path: str,
    frames_output_dir: Path,
    interval_seconds: float
) -> Dict[str, Any]:
    """
    Extrai frames do vídeo.
    """
    return extract_frames(
        video_path=video_path,
        output_dir=str(frames_output_dir),
        interval_seconds=interval_seconds
    )


def process_scenes(
    video_path: str,
    threshold: float,
    min_scene_gap_seconds: float
) -> Dict[str, Any]:
    """
    Detecta mudanças de cena no vídeo.
    """
    return detect_scene_changes(
        video_path=video_path,
        threshold=threshold,
        min_scene_gap_seconds=min_scene_gap_seconds
    )


def get_video_duration_from_metadata(metadata_result: Dict[str, Any]) -> float:
    """
    Tenta descobrir a duração do vídeo a partir dos metadados disponíveis.
    """
    possible_duration_keys = [
        "duration",
        "duration_seconds",
        "video_duration",
        "total_duration",
    ]

    for key in possible_duration_keys:
        value = metadata_result.get(key)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    fps = metadata_result.get("fps")
    total_frames = metadata_result.get("total_frames")

    if fps and total_frames:
        try:
            fps = float(fps)
            total_frames = float(total_frames)

            if fps > 0:
                return total_frames / fps
        except (TypeError, ValueError):
            pass

    return 0.0


def extract_timestamp_from_frame_name(frame_path: Path) -> Optional[float]:
    """
    Tenta extrair timestamp do nome do frame.

    Exemplos aceitos:
        frame_000001_t2.00.jpg
        frame_t12.50.jpg
    """
    name = frame_path.stem

    patterns = [
        r"_t(\d+(?:\.\d+)?)",
        r"t(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None

    return None


def normalize_frames_for_spectra(
    frames_result: Dict[str, Any],
    frame_interval_seconds: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Converte o retorno da extração de frames para o formato esperado pela Spectra.

    Formato esperado:
        [
            {
                "frame_path": "...",
                "timestamp": 0.0
            }
        ]
    """
    possible_keys = [
        "frames",
        "saved_frames",
        "extracted_frames",
        "frame_data",
        "frames_data",
    ]

    frames = None

    for key in possible_keys:
        if key in frames_result and isinstance(frames_result[key], list):
            frames = frames_result[key]
            break

    normalized_frames = []

    if frames is not None:
        for index, frame in enumerate(frames):
            if isinstance(frame, str):
                frame_path = frame
                timestamp = None
            elif isinstance(frame, dict):
                frame_path = (
                    frame.get("frame_path")
                    or frame.get("path")
                    or frame.get("file_path")
                    or frame.get("image_path")
                    or frame.get("output_path")
                )

                timestamp = (
                    frame.get("timestamp")
                    or frame.get("timestamp_seconds")
                    or frame.get("time")
                    or frame.get("second")
                    or frame.get("seconds")
                )
            else:
                continue

            if not frame_path:
                continue

            frame_path_obj = Path(frame_path)

            if timestamp is None:
                timestamp = extract_timestamp_from_frame_name(frame_path_obj)

            if timestamp is None:
                timestamp = index * frame_interval_seconds

            normalized_frames.append({
                "frame_path": str(frame_path_obj),
                "timestamp": float(timestamp),
            })

        return normalized_frames

    frames_output_dir = frames_result.get("frames_output_dir")

    if not frames_output_dir:
        return []

    frames_dir = Path(frames_output_dir)

    if not frames_dir.exists():
        return []

    frame_paths = sorted(
        list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.jpeg"))
        + list(frames_dir.glob("*.png"))
    )

    for index, frame_path in enumerate(frame_paths):
        timestamp = extract_timestamp_from_frame_name(frame_path)

        if timestamp is None:
            timestamp = index * frame_interval_seconds

        normalized_frames.append({
            "frame_path": str(frame_path),
            "timestamp": float(timestamp),
        })

    return normalized_frames


def normalize_scenes_for_spectra(
    scenes_result: Dict[str, Any],
    metadata_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Converte o retorno da detecção de cenas para o formato esperado pela Spectra.

    Formato esperado:
        [
            {
                "scene_id": 1,
                "start_time": 0.0,
                "end_time": 5.0
            }
        ]
    """
    if "scenes" in scenes_result and isinstance(scenes_result["scenes"], list):
        normalized_scenes = []

        for index, scene in enumerate(scenes_result["scenes"]):
            if not isinstance(scene, dict):
                continue

            start_time = (
                scene.get("start_time")
                or scene.get("start")
                or scene.get("start_time_seconds")
                or scene.get("start_seconds")
            )

            end_time = (
                scene.get("end_time")
                or scene.get("end")
                or scene.get("end_time_seconds")
                or scene.get("end_seconds")
            )

            if start_time is not None and end_time is not None:
                normalized_scenes.append({
                    "scene_id": scene.get("scene_id", index + 1),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                })

        if normalized_scenes:
            return normalized_scenes

    duration = get_video_duration_from_metadata(metadata_result)

    timestamps = scenes_result.get("scene_timestamps_seconds", [])

    if not timestamps:
        return [
            {
                "scene_id": 1,
                "start_time": 0.0,
                "end_time": float(duration),
            }
        ]

    clean_timestamps = []

    for value in timestamps:
        try:
            timestamp = float(value)

            if timestamp > 0:
                clean_timestamps.append(timestamp)
        except (TypeError, ValueError):
            continue

    clean_timestamps = sorted(set(clean_timestamps))

    scene_boundaries = [0.0] + clean_timestamps

    if duration > 0 and duration > scene_boundaries[-1]:
        scene_boundaries.append(float(duration))

    if len(scene_boundaries) == 1:
        scene_boundaries.append(float(duration))

    normalized_scenes = []

    for index in range(len(scene_boundaries) - 1):
        start_time = scene_boundaries[index]
        end_time = scene_boundaries[index + 1]

        if end_time <= start_time:
            continue

        normalized_scenes.append({
            "scene_id": index + 1,
            "start_time": float(start_time),
            "end_time": float(end_time),
        })

    if not normalized_scenes:
        normalized_scenes.append({
            "scene_id": 1,
            "start_time": 0.0,
            "end_time": float(duration),
        })

    return normalized_scenes


def save_json_result(result: Dict[str, Any], output_path: Path) -> None:
    """
    Salva um resultado em JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


def process_visual_analysis(
    frames_result: Dict[str, Any],
    scenes_result: Dict[str, Any],
    metadata_result: Dict[str, Any],
    visual_output_dir: Path,
    frame_interval_seconds: float,
    spectra_model_path: str,
    spectra_threshold: float,
    spectra_top_k: int,
) -> Dict[str, Any]:
    """
    Analisa visualmente os frames e cenas usando a Spectra.
    """
    model_path = Path(spectra_model_path)

    if not model_path.exists():
        return {
            "status": "skipped",
            "reason": f"Modelo da Spectra não encontrado: {spectra_model_path}",
            "frames_available_count": 0,
            "frames_analyzed_count": 0,
            "scenes_analyzed_count": 0,
            "scene_analyses": [],
        }

    frames_for_spectra = normalize_frames_for_spectra(
        frames_result=frames_result,
        frame_interval_seconds=frame_interval_seconds,
    )

    scenes_for_spectra = normalize_scenes_for_spectra(
        scenes_result=scenes_result,
        metadata_result=metadata_result,
    )

    if not frames_for_spectra:
        return {
            "status": "skipped",
            "reason": "Nenhum frame disponível para análise visual.",
            "frames_available_count": 0,
            "frames_analyzed_count": 0,
            "scenes_analyzed_count": 0,
            "scene_analyses": [],
        }

    spectra = Spectra(
        model_path=spectra_model_path,
        threshold=spectra_threshold,
        top_k=spectra_top_k,
    )

    visual_analysis = spectra.analyze_scenes(
        scenes=scenes_for_spectra,
        frames=frames_for_spectra,
    )

    visual_analysis["status"] = "completed"
    visual_analysis["frames_available_count"] = len(frames_for_spectra)
    visual_analysis["normalized_scenes_count"] = len(scenes_for_spectra)

    output_path = visual_output_dir / "visual_analysis_result.json"

    save_json_result(
        result=visual_analysis,
        output_path=output_path,
    )

    visual_analysis["visual_analysis_output_path"] = str(output_path)

    return visual_analysis


def build_processing_result(
    video_path: Path,
    metadata_result: Dict[str, Any],
    audio_result: Dict[str, Any],
    whisper_result: Dict[str, Any],
    frames_result: Dict[str, Any],
    scenes_result: Dict[str, Any],
    visual_analysis_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Monta o resultado final da pipeline.
    """
    return {
        "source_video_name": video_path.name,
        "source_video_path": str(video_path.resolve()),
        "metadata": metadata_result,
        "audio": {
            "extraction": audio_result,
            "speech_analysis": whisper_result,
        },
        "frames": frames_result,
        "scenes": scenes_result,
        "visual_analysis": visual_analysis_result,
    }


def process_video(
    video_path: str,
    output_base_dir: str = "data/output",
    frame_interval_seconds: float = 1.0,
    scene_threshold: float = 20.0,
    min_scene_gap_seconds: float = 0.5,

    whisper_model_size: str = "small",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    whisper_language: Optional[str] = None,
    whisper_beam_size: int = 5,
    whisper_vad_filter: bool = True,
    min_pause_duration: float = 0.5,
    refine_with_detected_language: bool = True,

    enable_visual_analysis: bool = True,
    spectra_model_path: str = "data/models/spectra/spectra_vision_net_best.pt",
    spectra_threshold: float = 0.4,
    spectra_top_k: int = 10,
) -> Dict[str, Any]:
    """
    Função principal da pipeline de vídeo.

    Etapas:
    - valida o vídeo
    - cria diretórios de saída
    - obtém metadados
    - extrai áudio
    - analisa fala e pausas com Whisper
    - extrai frames
    - detecta mudanças de cena
    - analisa frames/cenas com Spectra
    """
    validated_video_path = validate_video_path(video_path)
    validated_output_base_dir = ensure_output_base_directory(output_base_dir)
    output_dirs = build_output_directories(validated_output_base_dir)

    metadata_result = process_metadata(str(validated_video_path))

    audio_result = process_audio(
        str(validated_video_path),
        output_dirs["audio_dir"]
    )

    whisper_result = analyze_extracted_audio(
        audio_result=audio_result,
        whisper_model_size=whisper_model_size,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_language=whisper_language,
        whisper_beam_size=whisper_beam_size,
        whisper_vad_filter=whisper_vad_filter,
        min_pause_duration=min_pause_duration,
        refine_with_detected_language=refine_with_detected_language,
    )

    frames_result = process_frames(
        str(validated_video_path),
        output_dirs["frames_dir"],
        frame_interval_seconds
    )

    scenes_result = process_scenes(
        str(validated_video_path),
        scene_threshold,
        min_scene_gap_seconds
    )

    if enable_visual_analysis:
        visual_analysis_result = process_visual_analysis(
            frames_result=frames_result,
            scenes_result=scenes_result,
            metadata_result=metadata_result,
            visual_output_dir=output_dirs["visual_dir"],
            frame_interval_seconds=frame_interval_seconds,
            spectra_model_path=spectra_model_path,
            spectra_threshold=spectra_threshold,
            spectra_top_k=spectra_top_k,
        )
    else:
        visual_analysis_result = {
            "status": "disabled",
            "reason": "Análise visual desativada.",
            "frames_available_count": 0,
            "frames_analyzed_count": 0,
            "scenes_analyzed_count": 0,
            "scene_analyses": [],
        }

    return build_processing_result(
        video_path=validated_video_path,
        metadata_result=metadata_result,
        audio_result=audio_result,
        whisper_result=whisper_result,
        frames_result=frames_result,
        scenes_result=scenes_result,
        visual_analysis_result=visual_analysis_result,
    )