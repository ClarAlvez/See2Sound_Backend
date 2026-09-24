from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from ai.spectra.Person.person_cropper import PersonCropper


class CorrelationPersonAdapter:
    def __init__(
        self,
        cropper: Optional[PersonCropper] = None,
        person_predictor=None,
        max_people: int = 5,
    ):
        self.cropper = cropper or PersonCropper()

        self.person_predictor = person_predictor

        self.max_people = max_people

    def process_frame(
        self,
        frame_path: str,
        frame_id: int,
        timestamp: float,
        output_dir: str,
        scene_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        frame_path = Path(
            frame_path
        )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        crops = self.cropper.crop_people(
            image_path=str(
                frame_path
            ),
            output_dir=str(
                output_dir
            ),
            max_people=self.max_people,
        )

        people = []

        for index, crop in enumerate(
            crops
        ):
            attributes = (
                self.predict_person_attributes(
                    crop["crop_path"]
                )
            )
            
            people.append(
                {
                    "detection_id": (
                        f"frame_"
                        f"{frame_id:06d}_"
                        f"person_"
                        f"{index:02d}"
                    ),

                    "bbox": crop.get(
                        "detection_bbox",
                        crop["bbox"],
                    ),

                    "crop_bbox": crop["bbox"],

                    "crop_path": crop[
                        "crop_path"
                    ],

                    "detector_confidence": crop[
                        "detector_confidence"
                    ],

                    "attributes": attributes,
                }
            )

        return {
            "frame_id": frame_id,

            "timestamp": timestamp,

            "frame_path": str(
                frame_path
            ),

            "scene_id": scene_id,

            "people": people,
        }

    def predict_person_attributes(
        self,
        crop_path,
    ):
        if self.person_predictor is None:
            return {}

        labels = getattr(
            self.person_predictor,
            "labels",
            [],
        )

        top_k = max(
            1,
            len(labels),
        )

        result = (
            self.person_predictor.predict_top_labels(
                image_path=str(
                    crop_path
                ),

                top_k=top_k,
            )
        )

        predictions = result.get(
            "top_predictions",
            []
        )

        attributes = {}

        for prediction in predictions:
            label = prediction.get(
                "label"
            )

            score = prediction.get(
                "score"
            )

            if label is None or score is None:
                continue

            attributes[
                label
            ] = float(
                score
            )

        return attributes

    def process_frames(
        self,
        frames: List[Dict[str, Any]],
        output_dir: str,
    ) -> List[Dict[str, Any]]:

        results = []

        for index, frame in enumerate(
            frames
        ):
            frame_id = int(
                frame.get(
                    "frame_id",
                    index,
                )
            )

            timestamp = float(
                frame.get(
                    "timestamp",
                    0.0,
                )
            )

            frame_path = frame.get(
                "frame_path"
            )

            scene_id = frame.get(
                "scene_id"
            )

            if not frame_path:
                continue

            frame_output_dir = (
                Path(output_dir)
                / f"frame_{frame_id:06d}"
            )

            result = self.process_frame(
                frame_path=frame_path,
                frame_id=frame_id,
                timestamp=timestamp,
                output_dir=str(
                    frame_output_dir
                ),
                scene_id=scene_id,
            )

            results.append(
                result
            )

        return results


def infer_frame_timestamp(
    frame_path,
    index,
    frame_interval_seconds,
):
    frame_path = Path(
        frame_path
    )

    match = re.search(
        r"_t(\d+(?:\.\d+)?)",
        frame_path.stem,
    )

    if match:
        return float(
            match.group(1)
        )

    return (
        index
        * frame_interval_seconds
    )


def normalize_pipeline_frames(
    frames_result,
    frame_interval_seconds=0.5,
):
    raw_frames = []

    if isinstance(
        frames_result,
        list,
    ):
        raw_frames = frames_result

    elif isinstance(
        frames_result,
        dict,
    ):
        raw_frames = (
            frames_result.get(
                "frames"
            )
            or frames_result.get(
                "extracted_frames"
            )
            or []
        )

        if not raw_frames:
            frames_output_dir = (
                frames_result.get(
                    "frames_output_dir"
                )
            )

            if frames_output_dir:
                frames_dir = Path(
                    frames_output_dir
                )

                if not frames_dir.exists():
                    raise FileNotFoundError(
                        "Pasta de frames "
                        f"não encontrada: {frames_dir}"
                    )

                raw_frames = sorted(
                    list(
                        frames_dir.glob(
                            "*.jpg"
                        )
                    )
                    + list(
                        frames_dir.glob(
                            "*.jpeg"
                        )
                    )
                    + list(
                        frames_dir.glob(
                            "*.png"
                        )
                    )
                )

    else:
        raise ValueError(
            "Formato de frames_result "
            "não reconhecido."
        )

    normalized_frames = []

    for index, frame in enumerate(
        raw_frames
    ):
        scene_id = None

        if isinstance(
            frame,
            (str, Path),
        ):
            frame_path = str(
                frame
            )

            timestamp = (
                infer_frame_timestamp(
                    frame_path=frame_path,
                    index=index,
                    frame_interval_seconds=(
                        frame_interval_seconds
                    ),
                )
            )

        elif isinstance(
            frame,
            dict,
        ):
            frame_path = (
                frame.get(
                    "frame_path"
                )
                or frame.get(
                    "path"
                )
                or frame.get(
                    "file_path"
                )
            )

            if not frame_path:
                continue

            timestamp = frame.get(
                "timestamp"
            )

            if timestamp is None:
                timestamp = (
                    infer_frame_timestamp(
                        frame_path=frame_path,
                        index=index,
                        frame_interval_seconds=(
                            frame_interval_seconds
                        ),
                    )
                )

            timestamp = float(
                timestamp
            )

            scene_id = frame.get(
                "scene_id"
            )

        else:
            continue

        normalized_frames.append(
            {
                "frame_id": index,
                "timestamp": timestamp,
                "frame_path": str(
                    frame_path
                ),
                "scene_id": scene_id,
            }
        )

    if not normalized_frames:
        raise ValueError(
            "Nenhum frame pôde ser "
            "normalizado para a Correlation."
        )

    return normalized_frames