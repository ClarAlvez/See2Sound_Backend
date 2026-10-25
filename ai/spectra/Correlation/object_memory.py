from typing import Dict

from ai.spectra.Correlation.config import (
    CorrelationConfig,
)
from ai.spectra.Correlation.object_entities import (
    ObjectDetection,
    ObjectEntity,
)


class ObjectMemory:
    def __init__(
        self,
        config: CorrelationConfig,
    ):
        self.config = config

        self.objects: Dict[
            str,
            ObjectEntity,
        ] = {}

        self.next_object_id = 1

    def create_object(
        self,
        detection: ObjectDetection,
    ):
        object_id = (
            f"object_"
            f"{self.next_object_id:04d}"
        )

        self.next_object_id += 1

        entity = ObjectEntity(
            object_id=object_id,

            label=detection.label,

            first_seen=(
                detection.timestamp
            ),

            last_seen=(
                detection.timestamp
            ),

            first_frame=(
                detection.frame_id
            ),

            last_frame=(
                detection.frame_id
            ),
        )

        self.objects[
            object_id
        ] = entity

        self.update_object(
            object_id=object_id,
            detection=detection,
        )

        return object_id

    def update_object(
        self,
        object_id: str,
        detection: ObjectDetection,
    ):
        entity = self.objects[
            object_id
        ]

        entity.observation_count += 1

        entity.last_seen = max(
            entity.last_seen,
            detection.timestamp,
        )

        entity.last_frame = max(
            entity.last_frame,
            detection.frame_id,
        )

        entity.score_sum += (
            detection.label_score
        )

        entity.max_score = max(
            entity.max_score,
            detection.label_score,
        )

        entity.last_bbox = (
            detection.bbox
        )

        if (
            detection.frame_id
            not in entity.observation_frames
        ):
            entity.observation_frames.append(
                detection.frame_id
            )

        if detection.scene_id:
            entity.scenes_seen[
                detection.scene_id
            ] = (
                entity.scenes_seen.get(
                    detection.scene_id,
                    0,
                )
                + 1
            )

        if (
            detection.crop_path
            and detection.crop_path
            not in entity.example_crops
            and len(
                entity.example_crops
            )
            < self.config.max_example_crops_per_entity
        ):
            entity.example_crops.append(
                detection.crop_path
            )

    def register_track(
        self,
        object_id,
        track_id,
    ):
        entity = self.objects[
            object_id
        ]

        if (
            track_id
            not in entity.track_ids
        ):
            entity.track_ids.append(
                track_id
            )

    def get_object(
        self,
        object_id,
    ):
        return self.objects.get(
            object_id
        )

    def to_dict(self):
        return {
            object_id: entity.to_dict()
            for object_id, entity
            in self.objects.items()
        }