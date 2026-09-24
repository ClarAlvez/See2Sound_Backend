from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from ai.spectra.Correlation.config import CorrelationConfig
from ai.spectra.Correlation.entities import (
    PersonDetection,
    PersonEntity,
)
from ai.spectra.Correlation.similarity import (
    attribute_similarity,
    combine_scores,
    cosine_similarity,
    embedding_gallery_similarity,
    temporal_similarity,
)


class EntityMemory:
    def __init__(
        self,
        config: CorrelationConfig,
    ):
        self.config = config

        self.entities: Dict[
            str,
            PersonEntity,
        ] = {}

        self.next_person_id = 1

        self.last_reidentification_debug = {}

    def create_entity(
        self,
        detection: PersonDetection,
    ):
        entity_id = (
            f"person_{self.next_person_id:04d}"
        )

        self.next_person_id += 1

        entity = PersonEntity(
            entity_id=entity_id,

            first_seen=detection.timestamp,
            last_seen=detection.timestamp,

            first_frame=detection.frame_id,
            last_frame=detection.frame_id,
        )

        self.entities[
            entity_id
        ] = entity

        self.update_entity(
            entity_id=entity_id,
            detection=detection,
        )

        return entity_id

    def update_entity(
        self,
        entity_id: str,
        detection: PersonDetection,
    ):
        entity = self.entities[
            entity_id
        ]

        entity.observation_count += 1

        entity.last_seen = max(
            float(entity.last_seen),
            float(detection.timestamp),
        )

        entity.last_frame = max(
            int(entity.last_frame),
            int(detection.frame_id),
        )

        entity.first_seen = min(
            float(entity.first_seen),
            float(detection.timestamp),
        )

        entity.first_frame = min(
            int(entity.first_frame),
            int(detection.frame_id),
        )

        self.update_embedding(
            entity=entity,
            new_embedding=(
                detection.embedding
            ),
        )

        self.update_attributes(
            entity=entity,
            attributes=(
                detection.attributes
            ),
        )

        self.update_accessories(
            entity=entity,
            attributes=(
                detection.attributes
            ),
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

        if (
            detection.frame_id
            not in entity.observation_frames
        ):
            entity.observation_frames.append(
                int(
                    detection.frame_id
                )
            )

        entity.observation_timestamps.append(
            float(
                detection.timestamp
            )
        )

    # ============================================================
    # Embeddings
    # ============================================================

    def update_embedding(
        self,
        entity: PersonEntity,
        new_embedding,
    ):
        if new_embedding is None:
            return

        new_embedding = (
            new_embedding
            .float()
            .detach()
            .cpu()
            .flatten()
        )

        if new_embedding.numel() == 0:
            return

        new_embedding = F.normalize(
            new_embedding.unsqueeze(0),
            p=2,
            dim=1,
        ).squeeze(0)

        previous_count = (
            entity.embedding_count
        )

        if entity.embedding is None:
            entity.embedding = (
                new_embedding.clone()
            )

        else:
            combined = (
                entity.embedding
                * previous_count
                + new_embedding
            )

            entity.embedding = F.normalize(
                combined.unsqueeze(0),
                p=2,
                dim=1,
            ).squeeze(0)

        entity.embedding_count += 1

        self._add_embedding_to_history(
            entity=entity,
            embedding=new_embedding,
        )

    def _add_embedding_to_history(
        self,
        entity: PersonEntity,
        embedding: torch.Tensor,
    ):
        if not entity.embedding_history:
            entity.embedding_history.append(
                embedding.clone()
            )

            return

        similarities = []

        for previous_embedding in (
            entity.embedding_history
        ):
            score = cosine_similarity(
                embedding,
                previous_embedding,
            )

            if score is not None:
                similarities.append(
                    score
                )

        if similarities:
            most_similar = max(
                similarities
            )

            # Não desperdiça a galeria com várias
            # imagens quase idênticas.
            if (
                most_similar
                >= self.config.embedding_history_duplicate_threshold
            ):
                return

        entity.embedding_history.append(
            embedding.clone()
        )

        while (
            len(
                entity.embedding_history
            )
            > self.config.max_embedding_history
        ):
            entity.embedding_history.pop(
                0
            )

    # ============================================================
    # Atributos
    # ============================================================

    def update_attributes(
        self,
        entity: PersonEntity,
        attributes,
    ):
        if not attributes:
            return

        alpha = max(
            0.0,
            min(
                1.0,
                float(
                    self.config.recent_attribute_alpha
                ),
            ),
        )

        for label, score in attributes.items():
            score = max(
                0.0,
                min(
                    1.0,
                    float(score),
                ),
            )

            current_count = (
                entity.attribute_counts.get(
                    label,
                    0,
                )
            )

            current_average = (
                entity.attributes.get(
                    label,
                    0.0,
                )
            )

            new_average = (
                (
                    current_average
                    * current_count
                )
                + score
            ) / (
                current_count + 1
            )

            entity.attributes[
                label
            ] = new_average

            entity.attribute_counts[
                label
            ] = (
                current_count + 1
            )

            if (
                label
                not in entity.recent_attributes
            ):
                entity.recent_attributes[
                    label
                ] = score

            else:
                entity.recent_attributes[
                    label
                ] = (
                    alpha
                    * score
                    + (
                        1.0
                        - alpha
                    )
                    * entity.recent_attributes[
                        label
                    ]
                )

    def update_accessories(
        self,
        entity: PersonEntity,
        attributes,
    ):
        if not attributes:
            return

        for label in (
            self.config.accessory_attributes
        ):
            score = attributes.get(
                label
            )

            if score is None:
                continue

            if (
                float(score)
                < self.config.attribute_positive_threshold
            ):
                continue

            entity.accessory_counts[
                label
            ] = (
                entity.accessory_counts.get(
                    label,
                    0,
                )
                + 1
            )

    # ============================================================
    # Re-identificação online
    # ============================================================

    def find_reidentification(
        self,
        detection: PersonDetection,
        excluded_entity_ids: Optional[
            Iterable[str]
        ] = None,
    ) -> Tuple[
        Optional[str],
        float,
    ]:
        excluded_entity_ids = set(
            excluded_entity_ids or []
        )

        candidates = []

        for (
            entity_id,
            entity,
        ) in self.entities.items():

            if (
                entity_id
                in excluded_entity_ids
            ):
                continue

            gallery_score = (
                embedding_gallery_similarity(
                    embedding=(
                        detection.embedding
                    ),

                    embedding_history=(
                        entity.embedding_history
                    ),

                    top_k=(
                        self.config.gallery_top_k
                    ),
                )
            )

            centroid_score = (
                cosine_similarity(
                    detection.embedding,
                    entity.embedding,
                )
            )

            stable_attribute_score = (
                attribute_similarity(
                    detection.attributes,
                    entity.attributes,
                    self.config.stable_attributes,

                    evidence_threshold=(
                        self.config.attribute_evidence_threshold
                    ),
                )
            )

            appearance_attribute_score = (
                attribute_similarity(
                    detection.attributes,
                    entity.recent_attributes,
                    self.config.appearance_attributes,

                    evidence_threshold=(
                        self.config.attribute_evidence_threshold
                    ),
                )
            )

            temporal_score = (
                temporal_similarity(
                    current_timestamp=(
                        detection.timestamp
                    ),

                    last_seen=(
                        entity.last_seen
                    ),

                    window_seconds=(
                        self.config.temporal_reid_window_seconds
                    ),

                    floor=(
                        self.config.temporal_reid_floor
                    ),
                )
            )

            score = combine_scores([
                (
                    gallery_score,
                    self.config.reid_gallery_weight,
                ),

                (
                    centroid_score,
                    self.config.reid_centroid_weight,
                ),

                (
                    stable_attribute_score,
                    self.config.reid_stable_attribute_weight,
                ),

                (
                    appearance_attribute_score,
                    self.config.reid_appearance_attribute_weight,
                ),

                (
                    temporal_score,
                    self.config.reid_temporal_weight,
                ),
            ])

            candidates.append(
                {
                    "entity_id": entity_id,

                    "score": score,

                    "gallery_score": (
                        gallery_score
                    ),

                    "centroid_score": (
                        centroid_score
                    ),

                    "stable_attribute_score": (
                        stable_attribute_score
                    ),

                    "appearance_attribute_score": (
                        appearance_attribute_score
                    ),

                    "temporal_score": (
                        temporal_score
                    ),
                }
            )

        candidates.sort(
            key=lambda item: item[
                "score"
            ],
            reverse=True,
        )

        if not candidates:
            self.last_reidentification_debug = {
                "accepted": False,
                "reason": "no_candidates",
                "candidates": [],
            }

            return (
                None,
                0.0,
            )

        best = candidates[0]

        second_score = (
            candidates[1]["score"]
            if len(candidates) > 1
            else None
        )

        if second_score is None:
            margin = 1.0

        else:
            margin = (
                best["score"]
                - second_score
            )

        accepted = (
            best["score"]
            >= self.config.reidentification_min_score
            and margin
            >= self.config.reidentification_min_margin
        )

        self.last_reidentification_debug = {
            "accepted": accepted,

            "best_entity_id": best[
                "entity_id"
            ],

            "best_score": round(
                float(
                    best["score"]
                ),
                4,
            ),

            "second_score": (
                round(
                    float(second_score),
                    4,
                )
                if second_score is not None
                else None
            ),

            "margin": round(
                float(margin),
                4,
            ),

            "candidates": [
                self._serialize_candidate(
                    candidate
                )
                for candidate
                in candidates[:5]
            ],
        }

        if not accepted:
            return (
                None,
                float(
                    best["score"]
                ),
            )

        return (
            best["entity_id"],
            float(
                best["score"]
            ),
        )

    def _serialize_candidate(
        self,
        candidate,
    ):
        serialized = {}

        for key, value in (
            candidate.items()
        ):
            if isinstance(
                value,
                float,
            ):
                serialized[key] = round(
                    value,
                    4,
                )

            else:
                serialized[key] = value

        return serialized

    # ============================================================
    # Merge de entidades
    # ============================================================

    def merge_entities(
        self,
        target_entity_id: str,
        source_entity_id: str,
    ):
        if (
            target_entity_id
            == source_entity_id
        ):
            return target_entity_id

        target = self.entities.get(
            target_entity_id
        )

        source = self.entities.get(
            source_entity_id
        )

        if target is None:
            raise KeyError(
                "Entidade de destino não encontrada: "
                f"{target_entity_id}"
            )

        if source is None:
            raise KeyError(
                "Entidade de origem não encontrada: "
                f"{source_entity_id}"
            )

        target_last_seen_before_merge = (
            target.last_seen
        )

        source_last_seen_before_merge = (
            source.last_seen
        )

        self._merge_centroid(
            target=target,
            source=source,
        )

        for embedding in (
            source.embedding_history
        ):
            self._add_embedding_to_history(
                entity=target,
                embedding=embedding,
            )

        self._merge_attribute_averages(
            target=target,
            source=source,
        )

        # Aparência recente deve representar
        # o fragmento visto mais recentemente.
        if (
            source_last_seen_before_merge
            >= target_last_seen_before_merge
        ):
            target.recent_attributes = (
                source.recent_attributes.copy()
            )

        target.observation_count += (
            source.observation_count
        )

        target.first_seen = min(
            float(target.first_seen),
            float(source.first_seen),
        )

        target.last_seen = max(
            float(target.last_seen),
            float(source.last_seen),
        )

        target.first_frame = min(
            int(target.first_frame),
            int(source.first_frame),
        )

        target.last_frame = max(
            int(target.last_frame),
            int(source.last_frame),
        )

        for (
            label,
            count,
        ) in source.accessory_counts.items():

            target.accessory_counts[
                label
            ] = (
                target.accessory_counts.get(
                    label,
                    0,
                )
                + count
            )

        for (
            scene_id,
            count,
        ) in source.scenes_seen.items():

            target.scenes_seen[
                scene_id
            ] = (
                target.scenes_seen.get(
                    scene_id,
                    0,
                )
                + count
            )

        for crop_path in (
            source.example_crops
        ):
            if (
                crop_path
                not in target.example_crops
                and len(
                    target.example_crops
                )
                < self.config.max_example_crops_per_entity
            ):
                target.example_crops.append(
                    crop_path
                )

        for (
            alias,
            confidence,
        ) in source.aliases.items():

            target.aliases[
                alias
            ] = max(
                float(confidence),

                float(
                    target.aliases.get(
                        alias,
                        0.0,
                    )
                ),
            )
        
        for track_id in source.track_ids:
            if (
                track_id
                not in target.track_ids
            ):
                target.track_ids.append(
                    track_id
                )

        target.observation_frames = sorted(
            set(
                target.observation_frames
                + source.observation_frames
            )
        )

        target.observation_timestamps = sorted(
            target.observation_timestamps
            + source.observation_timestamps
        )

        del self.entities[
            source_entity_id
        ]

        return target_entity_id

    def _merge_centroid(
        self,
        target: PersonEntity,
        source: PersonEntity,
    ):
        target_count = (
            target.embedding_count
        )

        source_count = (
            source.embedding_count
        )

        if (
            target.embedding is None
            and source.embedding is not None
        ):
            target.embedding = (
                source.embedding.clone()
            )

            target.embedding_count = (
                source_count
            )

            return

        if (
            source.embedding is None
            or source_count <= 0
        ):
            return

        if (
            target.embedding is None
            or target_count <= 0
        ):
            target.embedding = (
                source.embedding.clone()
            )

            target.embedding_count = (
                source_count
            )

            return

        combined = (
            target.embedding
            * target_count
            + source.embedding
            * source_count
        )

        target.embedding = F.normalize(
            combined.unsqueeze(0),
            p=2,
            dim=1,
        ).squeeze(0)

        target.embedding_count = (
            target_count
            + source_count
        )

    def _merge_attribute_averages(
        self,
        target: PersonEntity,
        source: PersonEntity,
    ):
        labels = set(
            target.attribute_counts
        ) | set(
            source.attribute_counts
        )

        for label in labels:
            target_count = (
                target.attribute_counts.get(
                    label,
                    0,
                )
            )

            source_count = (
                source.attribute_counts.get(
                    label,
                    0,
                )
            )

            total_count = (
                target_count
                + source_count
            )

            if total_count <= 0:
                continue

            target_value = (
                target.attributes.get(
                    label,
                    0.0,
                )
            )

            source_value = (
                source.attributes.get(
                    label,
                    0.0,
                )
            )

            merged_value = (
                (
                    target_value
                    * target_count
                )
                + (
                    source_value
                    * source_count
                )
            ) / total_count

            target.attributes[
                label
            ] = merged_value

            target.attribute_counts[
                label
            ] = total_count

    def get_entity(
        self,
        entity_id,
    ):
        return self.entities.get(
            entity_id
        )

    def to_dict(self):
        return {
            entity_id: entity.to_dict()
            for entity_id, entity
            in self.entities.items()
        }

    def register_track(
        self,
        entity_id: str,
        track_id: str,
    ):
        entity = self.entities.get(
            entity_id
        )

        if entity is None:
            raise KeyError(
                "Entidade não encontrada: "
                f"{entity_id}"
            )

        if (
            track_id
            not in entity.track_ids
        ):
            entity.track_ids.append(
                track_id
            )