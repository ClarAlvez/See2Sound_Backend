import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.video.metadata import get_video_metadata
from pipeline.audio.extractor import extract_audio
from pipeline.audio.whisperer import WhisperPauseDetector
from pipeline.video.frames import extract_frames
from pipeline.video.scenes import detect_scene_changes

from ai.spectra.predictor import SpectraPredictor
from ai.spectra.Person.person_cropper import PersonCropper
try:
    from ai.spectra.Actions.person_action_analyzer import PersonActionAnalyzer
except ImportError:
    from ai.spectra.Actions.analyzer import PersonActionAnalyzer


# ============================================================
# Imports flexíveis para Narrative e TTS
# ============================================================

def create_narrative_generator(
    model_path: str = "data/models/llama/Llama-3.2-1B-Instruct-Q6_K_L.gguf",
):
    """
    Cria o gerador narrativo.

    Mantive import flexível porque o nome real do arquivo pode variar
    entre narrative_generator.py, generator.py etc.
    """
    try:
        from ai.narrative.narrative_generator import LLMNarrativeGenerator
    except ImportError:
        from ai.narrative.generator import LLMNarrativeGenerator

    return LLMNarrativeGenerator(
        model_path=model_path,
    )


def create_tts_engine(
    rate: int = 170,
    volume: float = 1.0,
):
    from ai.audio_description.tts_client import TTSClient

    return TTSClient(
        rate=rate,
        volume=volume,
    )


# ============================================================
# Validação e diretórios
# ============================================================

def validate_video_path(video_path: str) -> Path:
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    if not path.is_file():
        raise ValueError(f"O caminho informado não é um arquivo: {video_path}")

    return path


def ensure_output_base_directory(output_base_dir: str) -> Path:
    path = Path(output_base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_directories(output_base_dir: Path) -> Dict[str, Path]:
    audio_dir = output_base_dir / "audio"
    frames_dir = output_base_dir / "frames"
    scene_dir = output_base_dir / "scene_data"
    spectra_dir = output_base_dir / "spectra"
    person_crops_dir = output_base_dir / "person_crops"
    action_crops_dir = output_base_dir / "action_person_crops"
    narrative_dir = output_base_dir / "narrative"
    audio_description_dir = output_base_dir / "audio_descriptions"

    for directory in [
        audio_dir,
        frames_dir,
        scene_dir,
        spectra_dir,
        person_crops_dir,
        action_crops_dir,
        narrative_dir,
        audio_description_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "audio_dir": audio_dir,
        "frames_dir": frames_dir,
        "scene_dir": scene_dir,
        "spectra_dir": spectra_dir,
        "person_crops_dir": person_crops_dir,
        "action_crops_dir": action_crops_dir,
        "narrative_dir": narrative_dir,
        "audio_description_dir": audio_description_dir,
    }




def save_json(data: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    return output_path


# ============================================================
# Etapas originais
# ============================================================

def process_metadata(video_path: str) -> Dict[str, Any]:
    return get_video_metadata(video_path)


def process_audio(video_path: str, audio_output_dir: Path) -> Dict[str, Any]:
    return extract_audio(
        video_path=video_path,
        output_dir=str(audio_output_dir),
    )


def resolve_audio_path(audio_result: Dict[str, Any]) -> Path:
    possible_keys = [
        "audio_path",
        "audio_file_path",
        "output_audio_path",
        "path",
        "file_path",
    ]

    for key in possible_keys:
        value = audio_result.get(key)

        if not value:
            continue

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
    audio_path = resolve_audio_path(audio_result)

    detector = WhisperPauseDetector(
        model_size=whisper_model_size,
        device=whisper_device,
        compute_type=whisper_compute_type,
    )

    return detector.analyze_audio(
        audio_path=audio_path,
        language=whisper_language,
        beam_size=whisper_beam_size,
        vad_filter=whisper_vad_filter,
        min_pause_duration=min_pause_duration,
        refine_with_detected_language=refine_with_detected_language,
    )


def process_frames(
    video_path: str,
    frames_output_dir: Path,
    interval_seconds: float,
) -> Dict[str, Any]:
    return extract_frames(
        video_path=video_path,
        output_dir=str(frames_output_dir),
        interval_seconds=interval_seconds,
    )


def process_scenes(
    video_path: str,
    threshold: float,
    min_scene_gap_seconds: float,
) -> Dict[str, Any]:
    return detect_scene_changes(
        video_path=video_path,
        threshold=threshold,
        min_scene_gap_seconds=min_scene_gap_seconds,
    )


# ============================================================
# Frames e timestamps
# ============================================================

def collect_extracted_frames(frames_result: Dict[str, Any]) -> List[Path]:
    frames_output_dir = frames_result.get("frames_output_dir")

    if not frames_output_dir:
        raise ValueError(
            "frames_result não possui 'frames_output_dir'."
        )

    frames_dir = Path(frames_output_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Pasta de frames não encontrada: {frames_dir}"
        )

    frame_paths = sorted(
        list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.jpeg"))
        + list(frames_dir.glob("*.png"))
    )

    if not frame_paths:
        raise ValueError(
            f"Nenhum frame encontrado em: {frames_dir}"
        )

    return frame_paths


def infer_timestamp_from_frame_path(
    frame_path: Path,
    fallback_index: int,
    frame_interval_seconds: float,
) -> float:
    """
    Tenta extrair timestamp do nome do frame.

    Aceita nomes como:
    - frame_000001_t12.50.jpg
    - qualquer_nome_t12.50.jpg

    Se não encontrar, usa index * frame_interval_seconds.
    """
    match = re.search(r"_t(\d+(?:\.\d+)?)", frame_path.stem)

    if match:
        return float(match.group(1))

    return fallback_index * frame_interval_seconds


def build_frame_records(
    frame_paths: List[Path],
    frame_interval_seconds: float,
) -> List[Dict[str, Any]]:
    records = []

    for index, frame_path in enumerate(frame_paths):
        timestamp = infer_timestamp_from_frame_path(
            frame_path=frame_path,
            fallback_index=index,
            frame_interval_seconds=frame_interval_seconds,
        )

        records.append(
            {
                "index": index,
                "frame_path": frame_path,
                "timestamp": timestamp,
            }
        )

    return records


# ============================================================
# Cenas e intervalos
# ============================================================

def get_video_duration_seconds(metadata_result: Dict[str, Any]) -> Optional[float]:
    possible_keys = [
        "duration",
        "duration_seconds",
        "video_duration",
    ]

    for key in possible_keys:
        value = metadata_result.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    return None


def normalize_scene_timestamps(
    scenes_result: Dict[str, Any],
) -> List[float]:
    timestamps = scenes_result.get("scene_timestamps_seconds", [])

    normalized = []

    for value in timestamps:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            continue

    normalized = sorted(set(normalized))

    return normalized


def build_scene_intervals(
    scenes_result: Dict[str, Any],
    metadata_result: Dict[str, Any],
    fallback_end_time: float,
) -> List[Dict[str, float]]:
    scene_timestamps = normalize_scene_timestamps(scenes_result)
    video_duration = get_video_duration_seconds(metadata_result)

    if video_duration is None:
        video_duration = fallback_end_time

    if video_duration <= 0:
        video_duration = fallback_end_time

    # Garante começo em 0.
    boundaries = [0.0]

    for timestamp in scene_timestamps:
        if 0.0 < timestamp < video_duration:
            boundaries.append(timestamp)

    boundaries.append(video_duration)
    boundaries = sorted(set(boundaries))

    intervals = []

    for index in range(len(boundaries) - 1):
        start_time = boundaries[index]
        end_time = boundaries[index + 1]

        if end_time <= start_time:
            continue

        intervals.append(
            {
                "scene_index": index,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    if not intervals:
        intervals.append(
            {
                "scene_index": 0,
                "start_time": 0.0,
                "end_time": video_duration,
            }
        )

    return intervals


def select_frame_for_interval(
    frame_records: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
) -> Optional[Dict[str, Any]]:
    midpoint = (start_time + end_time) / 2

    candidates = [
        record
        for record in frame_records
        if start_time <= record["timestamp"] <= end_time
    ]

    if not candidates:
        candidates = frame_records

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda record: abs(record["timestamp"] - midpoint),
    )


# ============================================================
# Pausas de fala / inserção sugerida
# ============================================================

def extract_speech_pauses(whisper_result: Dict[str, Any]) -> List[Dict[str, float]]:
    pauses = whisper_result.get("speech_pauses", [])

    normalized = []

    for pause in pauses:
        try:
            start = float(pause.get("start"))
            end = float(pause.get("end"))
        except (TypeError, ValueError, AttributeError):
            continue

        if end <= start:
            continue

        normalized.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
            }
        )

    return normalized


def find_best_pause_for_interval(
    pauses: List[Dict[str, float]],
    start_time: float,
    end_time: float,
) -> Optional[Dict[str, float]]:
    overlapping = []

    for pause in pauses:
        overlap_start = max(start_time, pause["start"])
        overlap_end = min(end_time, pause["end"])
        overlap_duration = overlap_end - overlap_start

        if overlap_duration > 0:
            overlapping.append(
                {
                    **pause,
                    "overlap_duration": overlap_duration,
                }
            )

    if overlapping:
        return max(
            overlapping,
            key=lambda item: item["overlap_duration"],
        )

    # Se não houver pausa dentro da cena, usa a pausa mais próxima do início da cena.
    if not pauses:
        return None

    return min(
        pauses,
        key=lambda pause: abs(pause["start"] - start_time),
    )


# ============================================================
# Spectra: carregar modelos
# ============================================================

def maybe_create_predictor(
    model_path: Optional[str],
    threshold: float,
    top_k: int,
    task_name: str,
    strict: bool = False,
) -> Optional[SpectraPredictor]:
    if not model_path:
        return None

    path = Path(model_path)

    if not path.exists():
        if strict:
            raise FileNotFoundError(
                f"Modelo {task_name} não encontrado: {model_path}"
            )

        print(f"[Spectra] Modelo {task_name} não encontrado. Pulando: {model_path}")
        return None

    return SpectraPredictor(
        model_path=str(path),
        threshold=threshold,
        top_k=top_k,
        task_name=task_name,
    )

def create_action_analyzer(
    action_model_path: Optional[str],
    action_threshold: float = 0.3,
    action_top_k: int = 10,
    person_cropper_model_name: str = "yolov8n.pt",
    person_cropper_confidence_threshold: float = 0.35,
    max_people: int = 5,
):
    if action_model_path is None:
        return None

    action_model = Path(action_model_path)

    if not action_model.exists():
        print(f"Modelo Actions não encontrado, pulando Actions: {action_model}")
        return None

    return PersonActionAnalyzer(
        action_model_path=str(action_model),
        action_threshold=action_threshold,
        action_top_k=action_top_k,
        person_cropper_model_name=person_cropper_model_name,
        person_cropper_confidence_threshold=person_cropper_confidence_threshold,
        max_people=max_people,
    )
    
def analyze_frame_actions_with_person_crops(
    frame_path: str,
    action_analyzer,
    action_crops_dir: Path,
    threshold: float = 0.3,
    top_k: int = 10,
):
    if action_analyzer is None:
        return {
            "frame_path": str(frame_path),
            "task_name": "action",
            "source": "disabled",
            "predictions": [],
            "grouped_predictions": {
                "scene": [],
                "person": [],
                "object": [],
                "action": [],
            },
        }

    frame_path = Path(frame_path)

    frame_crops_dir = action_crops_dir / frame_path.stem
    frame_crops_dir.mkdir(parents=True, exist_ok=True)

    return action_analyzer.analyze_frame(
        image_path=str(frame_path),
        crops_output_dir=str(frame_crops_dir),
        threshold=threshold,
        top_k=top_k,
    )

def create_spectra_predictors(
    scene_model_path: Optional[str],
    person_model_path: Optional[str],
    object_model_path: Optional[str],
    scene_threshold: float,
    person_threshold: float,
    object_threshold: float,
    top_k: int,
    strict_model_loading: bool = False,
) -> Dict[str, Optional[SpectraPredictor]]:
    scene_predictor = maybe_create_predictor(
        model_path=scene_model_path,
        threshold=scene_threshold,
        top_k=top_k,
        task_name="scene",
        strict=strict_model_loading,
    )

    person_predictor = maybe_create_predictor(
        model_path=person_model_path,
        threshold=person_threshold,
        top_k=top_k,
        task_name="person",
        strict=strict_model_loading,
    )

    object_predictor = maybe_create_predictor(
        model_path=object_model_path,
        threshold=object_threshold,
        top_k=top_k,
        task_name="object",
        strict=strict_model_loading,
    )

    return {
        "scene": scene_predictor,
        "person": person_predictor,
        "object": object_predictor,
    }


# ============================================================
# Spectra: unificar outputs
# ============================================================

def extract_predictions_from_result(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not result:
        return []

    predictions = result.get("predictions", [])

    valid_predictions = []

    for prediction in predictions:
        label = prediction.get("label")
        score = prediction.get("score")

        if not label:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        valid_predictions.append(
            {
                "label": label,
                "score": score,
            }
        )

    return valid_predictions


def merge_spectra_predictions(
    scene_result: Optional[Dict[str, Any]],
    person_result: Optional[Dict[str, Any]],
    object_result: Optional[Dict[str, Any]],
    action_result: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, float], Dict[str, Any]]:
    task_results = {
        "scene": extract_predictions_from_result(scene_result),
        "person": extract_predictions_from_result(person_result),
        "object": extract_predictions_from_result(object_result),
        "action": extract_predictions_from_result(action_result),
    }

    label_scores: Dict[str, float] = {}
    confidence: Dict[str, float] = {}

    for task_name, predictions in task_results.items():
        for prediction in predictions:
            label = prediction["label"]
            score = prediction["score"]

            confidence[f"{task_name}.{label}"] = score

            if label not in label_scores:
                label_scores[label] = score
            else:
                label_scores[label] = max(label_scores[label], score)

    labels = [
        label
        for label, _ in sorted(
            label_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    context = {
        "spectra_tasks": task_results,
        "merged_label_scores": label_scores,
    }

    return labels, confidence, context




def merge_person_crop_results(
    crop_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    merged_scores: Dict[str, float] = {}

    raw_crop_results = []

    for item in crop_results:
        crop = item["crop"]
        result = item["result"]

        raw_crop_results.append(item)

        predictions = extract_predictions_from_result(result)

        for prediction in predictions:
            label = prediction["label"]
            score = prediction["score"]

            if label not in merged_scores:
                merged_scores[label] = score
            else:
                merged_scores[label] = max(merged_scores[label], score)

    predictions = [
        {
            "label": label,
            "score": score,
        }
        for label, score in sorted(
            merged_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    return {
        "labels": [prediction["label"] for prediction in predictions],
        "predictions": predictions,
        "raw_crop_results": raw_crop_results,
    }

def analyze_frame_with_spectra(
    frame_path: Path,
    predictors: Dict[str, Optional[SpectraPredictor]],
    use_person_model_on_full_frame: bool = False,
    use_object_model_on_full_frame: bool = False,
    person_cropper: Optional[PersonCropper] = None,
    person_crops_output_dir: Optional[Path] = None,
    action_analyzer: Optional[PersonActionAnalyzer] = None,
    action_crops_output_dir: Optional[Path] = None,
    spectra_action_threshold: float = 0.3,
    spectra_top_k: int = 10,
) -> Dict[str, Any]:
    scene_result = None
    person_result = None
    object_result = None
    action_result = None

    if predictors.get("scene") is not None:
        scene_result = predictors["scene"].predict_frame(
            image_path=str(frame_path),
            group_by_category=True,
        )

    person_crops = []

    if predictors.get("person") is not None:
        if person_cropper is not None and person_crops_output_dir is not None:
            person_frame_crops_dir = person_crops_output_dir / frame_path.stem
            person_frame_crops_dir.mkdir(parents=True, exist_ok=True)

            person_crops = person_cropper.crop_people(
                image_path=str(frame_path),
                output_dir=str(person_frame_crops_dir),
                max_people=3,
            )

            crop_results = []

            for crop in person_crops:
                crop_path = crop.get("crop_path") if isinstance(crop, dict) else str(crop)

                crop_result = predictors["person"].predict_frame(
                    image_path=crop_path,
                    group_by_category=True,
                )

                crop_results.append(
                    {
                        "crop": crop,
                        "result": crop_result,
                    }
                )

            if crop_results:
                person_result = merge_person_crop_results(crop_results)

        elif use_person_model_on_full_frame:
            person_result = predictors["person"].predict_frame(
                image_path=str(frame_path),
                group_by_category=True,
            )

    if predictors.get("object") is not None and use_object_model_on_full_frame:
        object_result = predictors["object"].predict_frame(
            image_path=str(frame_path),
            group_by_category=True,
        )

    if action_analyzer is not None and action_crops_output_dir is not None:
        action_result = analyze_frame_actions_with_person_crops(
            frame_path=str(frame_path),
            action_analyzer=action_analyzer,
            action_crops_dir=action_crops_output_dir,
            threshold=spectra_action_threshold,
            top_k=spectra_top_k,
        )

    labels, confidence, context = merge_spectra_predictions(
        scene_result=scene_result,
        person_result=person_result,
        object_result=object_result,
        action_result=action_result,
    )

    return {
        "frame_path": str(frame_path),
        "labels": labels,
        "confidence": confidence,
        "context": context,
        "raw": {
            "scene": scene_result,
            "person": person_result,
            "object": object_result,
            "action": action_result,
            "person_crops": person_crops,
        },
    }




def build_spectra_outputs_for_scenes(
    frame_records: List[Dict[str, Any]],
    scene_intervals: List[Dict[str, float]],
    whisper_result: Dict[str, Any],
    predictors: Dict[str, Optional[SpectraPredictor]],
    use_person_model_on_full_frame: bool = False,
    use_object_model_on_full_frame: bool = False,
    person_cropper: Optional[PersonCropper] = None,
    person_crops_output_dir: Optional[Path] = None,
    action_analyzer: Optional[PersonActionAnalyzer] = None,
    action_crops_output_dir: Optional[Path] = None,
    spectra_action_threshold: float = 0.3,
    spectra_top_k: int = 10,
) -> List[Dict[str, Any]]:
    pauses = extract_speech_pauses(whisper_result)

    spectra_outputs = []

    for interval in scene_intervals:
        frame_record = select_frame_for_interval(
            frame_records=frame_records,
            start_time=interval["start_time"],
            end_time=interval["end_time"],
        )

        if frame_record is None:
            continue

        spectra_frame_result = analyze_frame_with_spectra(
            frame_path=frame_record["frame_path"],
            predictors=predictors,
            use_person_model_on_full_frame=use_person_model_on_full_frame,
            use_object_model_on_full_frame=use_object_model_on_full_frame,
            person_cropper=person_cropper,
            person_crops_output_dir=person_crops_output_dir,
            action_analyzer=action_analyzer,
            action_crops_output_dir=action_crops_output_dir,
            spectra_action_threshold=spectra_action_threshold,
            spectra_top_k=spectra_top_k,
        )

        best_pause = find_best_pause_for_interval(
            pauses=pauses,
            start_time=interval["start_time"],
            end_time=interval["end_time"],
        )

        context = dict(spectra_frame_result["context"])
        context["frame_path"] = spectra_frame_result["frame_path"]
        context["scene_index"] = interval["scene_index"]
        context["selected_frame_timestamp"] = frame_record["timestamp"]
        context["raw_outputs"] = spectra_frame_result.get("raw", {})

        if best_pause:
            context["suggested_pause"] = best_pause

        spectra_outputs.append(
            {
                "start_time": interval["start_time"],
                "end_time": interval["end_time"],
                "labels": spectra_frame_result["labels"],
                "confidence": spectra_frame_result["confidence"],
                "context": context,
            }
        )

    return spectra_outputs




# ============================================================
# Narrative
# ============================================================

def generate_narrative_timeline(
    spectra_outputs: List[Dict[str, Any]],
    narrative_model_path: str,
) -> List[Dict[str, Any]]:
    generator = create_narrative_generator(
        model_path=narrative_model_path,
    )

    return generator.generate_timeline_from_dicts(
        spectra_outputs=spectra_outputs,
    )


# ============================================================
# Audio Description / TTS
# ============================================================

def safe_time_for_filename(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0

    return f"{number:08.2f}".replace(".", "_")


def generate_audio_description_files(
    narrative_timeline: List[Dict[str, Any]],
    output_dir: Path,
    tts_rate: int = 170,
    tts_volume: float = 1.0,
) -> List[Dict[str, Any]]:
    tts = create_tts_engine(
        rate=tts_rate,
        volume=tts_volume,
    )

    audio_outputs = []

    for index, item in enumerate(narrative_timeline):
        description = item.get("description", "")

        if not description:
            continue

        start_time = item.get("start_time", 0.0)
        end_time = item.get("end_time", start_time)

        file_name = "ad_{:04d}_{}_{}.wav".format(
            index,
            safe_time_for_filename(start_time),
            safe_time_for_filename(end_time),
        )

        output_path = output_dir / file_name

        saved_output = tts.save_to_file(
            text=description,
            output_path=str(output_path),
        )

        audio_outputs.append(
            {
                "index": index,
                "start_time": start_time,
                "end_time": end_time,
                "description": description,
                "audio_path": str(saved_output or output_path),
                "source_narrative": item,
            }
        )

    return audio_outputs


# ============================================================
# Resultado final
# ============================================================

def build_processing_result(
    video_path: Path,
    metadata_result: Dict[str, Any],
    audio_result: Dict[str, Any],
    whisper_result: Dict[str, Any],
    frames_result: Dict[str, Any],
    scenes_result: Dict[str, Any],
    spectra_outputs: List[Dict[str, Any]],
    narrative_timeline: List[Dict[str, Any]],
    audio_description_outputs: List[Dict[str, Any]],
    saved_artifacts: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "source_video_name": video_path.name,
        "source_video_path": str(video_path.resolve()),
        "metadata": metadata_result,
        "audio": {
            "extraction": audio_result,
            "speech_analysis": whisper_result,
            "description_generation": audio_description_outputs,
        },
        "frames": frames_result,
        "scenes": scenes_result,
        "spectra": {
            "outputs": spectra_outputs,
        },
        "narrative": {
            "timeline": narrative_timeline,
        },
        "artifacts": saved_artifacts,
    }


# ============================================================
# Pipeline principal
# ============================================================

def process_video(
    video_path: str,
    output_base_dir: str = "data/output",

    # Vídeo / cenas / frames
    frame_interval_seconds: float = 1.0,
    scene_threshold: float = 20.0,
    min_scene_gap_seconds: float = 0.5,

    # Whisper
    whisper_model_size: str = "small",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    whisper_language: Optional[str] = None,
    whisper_beam_size: int = 5,
    whisper_vad_filter: bool = True,
    min_pause_duration: float = 0.5,
    refine_with_detected_language: bool = True,

    # Spectra
    scene_model_path: Optional[str] = "data/models/spectra_scene/scene_net_best.pt",
    person_model_path: Optional[str] = None,
    object_model_path: Optional[str] = None,
    action_model_path: Optional[str] = None,
    spectra_scene_threshold: float = 0.45,
    spectra_person_threshold: float = 0.50,
    spectra_object_threshold: float = 0.50,
    spectra_action_threshold: float = 0.30,
    spectra_top_k: int = 12,
    strict_model_loading: bool = False,

    use_person_model_on_full_frame: bool = False,
    use_object_model_on_full_frame: bool = False,
    use_person_cropper: bool = True,
    person_cropper_model_name: str = "yolov8n.pt",
    person_cropper_confidence_threshold: float = 0.35,

    use_action_model: bool = True,
    use_action_person_cropper: bool = True,
    action_max_people: int = 5,

    # Narrative
    narrative_model_path: str = "data/models/llama/Llama-3.2-1B-Instruct-Q6_K_L.gguf",

    # TTS
    tts_rate: int = 170,
    tts_volume: float = 1.0,

    # Controle
    run_spectra: bool = True,
    run_narrative: bool = True,
    run_tts: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline final See2Sound.

    Etapas:
    1. Valida vídeo.
    2. Extrai metadata.
    3. Extrai áudio.
    4. Analisa fala/pausas com Whisper.
    5. Extrai frames.
    6. Detecta cenas.
    7. Executa Spectra Scene, Person, Object e Actions.
    8. Unifica labels da Spectra.
    9. Envia labels para Narrative.
    10. Gera arquivos de áudio com TTS.
    """
    validated_video_path = validate_video_path(video_path)
    validated_output_base_dir = ensure_output_base_directory(output_base_dir)
    output_dirs = build_output_directories(validated_output_base_dir)

    metadata_result = process_metadata(
        str(validated_video_path)
    )

    audio_result = process_audio(
        video_path=str(validated_video_path),
        audio_output_dir=output_dirs["audio_dir"],
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
        video_path=str(validated_video_path),
        frames_output_dir=output_dirs["frames_dir"],
        interval_seconds=frame_interval_seconds,
    )

    scenes_result = process_scenes(
        video_path=str(validated_video_path),
        threshold=scene_threshold,
        min_scene_gap_seconds=min_scene_gap_seconds,
    )

    frame_paths = collect_extracted_frames(frames_result)
    frame_records = build_frame_records(
        frame_paths=frame_paths,
        frame_interval_seconds=frame_interval_seconds,
    )

    fallback_end_time = 0.0

    if frame_records:
        fallback_end_time = frame_records[-1]["timestamp"] + frame_interval_seconds

    scene_intervals = build_scene_intervals(
        scenes_result=scenes_result,
        metadata_result=metadata_result,
        fallback_end_time=fallback_end_time,
    )

    spectra_outputs: List[Dict[str, Any]] = []

    person_cropper = None
    action_analyzer = None

    if run_spectra and use_person_cropper:
        person_cropper = PersonCropper(
            model_name=person_cropper_model_name,
            confidence_threshold=person_cropper_confidence_threshold,
        )

    if run_spectra and use_action_model and use_action_person_cropper:
        action_analyzer = create_action_analyzer(
            action_model_path=action_model_path,
            action_threshold=spectra_action_threshold,
            action_top_k=spectra_top_k,
            person_cropper_model_name=person_cropper_model_name,
            person_cropper_confidence_threshold=person_cropper_confidence_threshold,
            max_people=action_max_people,
        )

    if run_spectra:
        predictors = create_spectra_predictors(
            scene_model_path=scene_model_path,
            person_model_path=person_model_path,
            object_model_path=object_model_path,
            scene_threshold=spectra_scene_threshold,
            person_threshold=spectra_person_threshold,
            object_threshold=spectra_object_threshold,
            top_k=spectra_top_k,
            strict_model_loading=strict_model_loading,
        )

        if not any(predictors.values()) and action_analyzer is None:
            print("[Spectra] Nenhum modelo visual carregado. Pulando análise visual.")
            spectra_outputs = []
        else:
            spectra_outputs = build_spectra_outputs_for_scenes(
                frame_records=frame_records,
                scene_intervals=scene_intervals,
                whisper_result=whisper_result,
                predictors=predictors,
                use_person_model_on_full_frame=use_person_model_on_full_frame,
                use_object_model_on_full_frame=use_object_model_on_full_frame,
                person_cropper=person_cropper,
                person_crops_output_dir=output_dirs["person_crops_dir"],
                action_analyzer=action_analyzer,
                action_crops_output_dir=output_dirs["action_crops_dir"],
                spectra_action_threshold=spectra_action_threshold,
                spectra_top_k=spectra_top_k,
            )

    narrative_timeline: List[Dict[str, Any]] = []

    if run_narrative:
        narrative_timeline = generate_narrative_timeline(
            spectra_outputs=spectra_outputs,
            narrative_model_path=narrative_model_path,
        )

    audio_description_outputs: List[Dict[str, Any]] = []

    if run_tts:
        audio_description_outputs = generate_audio_description_files(
            narrative_timeline=narrative_timeline,
            output_dir=output_dirs["audio_description_dir"],
            tts_rate=tts_rate,
            tts_volume=tts_volume,
        )

    spectra_json_path = save_json(
        data=spectra_outputs,
        output_path=output_dirs["spectra_dir"] / "spectra_outputs.json",
    )

    narrative_json_path = save_json(
        data=narrative_timeline,
        output_path=output_dirs["narrative_dir"] / "narrative_timeline.json",
    )

    audio_descriptions_json_path = save_json(
        data=audio_description_outputs,
        output_path=output_dirs["audio_description_dir"] / "audio_description_outputs.json",
    )

    saved_artifacts = {
        "spectra_outputs_json": str(spectra_json_path),
        "narrative_timeline_json": str(narrative_json_path),
        "audio_description_outputs_json": str(audio_descriptions_json_path),
    }

    return build_processing_result(
        video_path=validated_video_path,
        metadata_result=metadata_result,
        audio_result=audio_result,
        whisper_result=whisper_result,
        frames_result=frames_result,
        scenes_result=scenes_result,
        spectra_outputs=spectra_outputs,
        narrative_timeline=narrative_timeline,
        audio_description_outputs=audio_description_outputs,
        saved_artifacts=saved_artifacts,
    )