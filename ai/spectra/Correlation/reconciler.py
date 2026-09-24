from typing import Dict, List, Optional, Tuple

from ai.spectra.Correlation.config import CorrelationConfig
from ai.spectra.Correlation.memory import EntityMemory
from ai.spectra.Correlation.similarity import (
    attribute_similarity,
    combine_scores,
    cosine_similarity,
    cross_gallery_similarity,
    temporal_interval_similarity,
)


class EntityReconciler:
    def __init__(
        self,
        memory: EntityMemory,
        config: CorrelationConfig,
    ):
        self.memory = memory
        self.config = config

    def reconcile(
        self,
        processed_frames: List[Dict],
    ):
        if not self.config.enable_reconciliation:
            return {
                "enabled": False,
                "merge_count": 0,
                "merges": [],
                "entity_remap": {},
            }

        merges = []
        entity_remap = {}

        while True:
            proposals = (
                self._build_merge_proposals()
            )

            if not proposals:
                break

            proposal = max(
                proposals,
                key=lambda item: item[
                    "score"
                ],
            )

            entity_a_id = proposal[
                "entity_a"
            ]

            entity_b_id = proposal[
                "entity_b"
            ]

            entity_a = (
                self.memory.get_entity(
                    entity_a_id
                )
            )

            entity_b = (
                self.memory.get_entity(
                    entity_b_id
                )
            )

            if (
                entity_a is None
                or entity_b is None
            ):
                continue

            # Preserva preferencialmente o ID
            # que apareceu primeiro no vídeo.
            if (
                entity_a.first_seen
                <= entity_b.first_seen
            ):
                target_entity_id = (
                    entity_a_id
                )

                source_entity_id = (
                    entity_b_id
                )

            else:
                target_entity_id = (
                    entity_b_id
                )

                source_entity_id = (
                    entity_a_id
                )

            self.memory.merge_entities(
                target_entity_id=(
                    target_entity_id
                ),

                source_entity_id=(
                    source_entity_id
                ),
            )

            entity_remap[
                source_entity_id
            ] = target_entity_id

            merges.append(
                {
                    "source_entity_id": (
                        source_entity_id
                    ),

                    "target_entity_id": (
                        target_entity_id
                    ),

                    "score": round(
                        float(
                            proposal["score"]
                        ),
                        4,
                    ),

                    "details": (
                        self._serialize_details(
                            proposal["details"]
                        )
                    ),
                }
            )

        final_remap = (
            self._resolve_entity_remap(
                entity_remap
            )
        )

        self._apply_remap_to_frames(
            processed_frames=(
                processed_frames
            ),

            entity_remap=(
                final_remap
            ),
        )

        return {
            "enabled": True,

            "merge_count": len(
                merges
            ),

            "merges": merges,

            "entity_remap": (
                final_remap
            ),
        }

    # ============================================================
    # Busca conservadora por merges
    # ============================================================

    def _build_merge_proposals(
    self,
    ):
        entity_ids = list(
            self.memory.entities.keys()
        )

        proposals = []

        for index_a in range(
            len(entity_ids)
        ):
            entity_a_id = (
                entity_ids[index_a]
            )

            for index_b in range(
                index_a + 1,
                len(entity_ids),
            ):
                entity_b_id = (
                    entity_ids[index_b]
                )

                if self._cooccur_in_same_frame(
                    entity_a_id,
                    entity_b_id,
                ):
                    continue

                (
                    score,
                    details,
                ) = self._score_pair(
                    entity_a_id,
                    entity_b_id,
                )

                if (
                    score
                    < self.config.reconcile_min_score
                ):
                    continue

                proposals.append(
                    {
                        "entity_a": (
                            entity_a_id
                        ),

                        "entity_b": (
                            entity_b_id
                        ),

                        "score": score,

                        "details": details,
                    }
                )

        proposals.sort(
            key=lambda item: item[
                "score"
            ],
            reverse=True,
        )

        return proposals

    # ============================================================
    # Score de duas entidades completas
    # ============================================================

    def _score_pair(
        self,
        entity_a_id: str,
        entity_b_id: str,
    ) -> Tuple[
        float,
        Dict[str, Optional[float]],
    ]:
        entity_a = (
            self.memory.get_entity(
                entity_a_id
            )
        )

        entity_b = (
            self.memory.get_entity(
                entity_b_id
            )
        )

        if (
            entity_a is None
            or entity_b is None
        ):
            return (
                0.0,
                {},
            )

        gallery_score = (
            cross_gallery_similarity(
                history_a=(
                    entity_a.embedding_history
                ),

                history_b=(
                    entity_b.embedding_history
                ),

                top_k=(
                    self.config.reconcile_gallery_top_k
                ),
            )
        )

        centroid_score = (
            cosine_similarity(
                entity_a.embedding,
                entity_b.embedding,
            )
        )

        stable_attribute_score = (
            attribute_similarity(
                entity_a.attributes,
                entity_b.attributes,
                self.config.stable_attributes,

                evidence_threshold=(
                    self.config.attribute_evidence_threshold
                ),
            )
        )

        appearance_attribute_score = (
            attribute_similarity(
                entity_a.recent_attributes,
                entity_b.recent_attributes,
                self.config.appearance_attributes,

                evidence_threshold=(
                    self.config.attribute_evidence_threshold
                ),
            )
        )

        temporal_score = (
            temporal_interval_similarity(
                first_a=(
                    entity_a.first_seen
                ),

                last_a=(
                    entity_a.last_seen
                ),

                first_b=(
                    entity_b.first_seen
                ),

                last_b=(
                    entity_b.last_seen
                ),

                window_seconds=(
                    self.config.reconcile_temporal_window_seconds
                ),

                floor=(
                    self.config.reconcile_temporal_floor
                ),
            )
        )

        score = combine_scores([
            (
                gallery_score,
                self.config.reconcile_gallery_weight,
            ),

            (
                centroid_score,
                self.config.reconcile_centroid_weight,
            ),

            (
                stable_attribute_score,
                self.config.reconcile_stable_attribute_weight,
            ),

            (
                appearance_attribute_score,
                self.config.reconcile_appearance_attribute_weight,
            ),

            (
                temporal_score,
                self.config.reconcile_temporal_weight,
            ),
        ])

        return (
            score,

            {
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
            },
        )

    # ============================================================
    # Segurança contra merge de pessoas simultâneas
    # ============================================================

    def _cooccur_in_same_frame(
        self,
        entity_a_id,
        entity_b_id,
    ):
        entity_a = (
            self.memory.get_entity(
                entity_a_id
            )
        )

        entity_b = (
            self.memory.get_entity(
                entity_b_id
            )
        )

        if (
            entity_a is None
            or entity_b is None
        ):
            return False

        frames_a = set(
            entity_a.observation_frames
        )

        frames_b = set(
            entity_b.observation_frames
        )

        return bool(
            frames_a
            & frames_b
        )

    # ============================================================
    # Remapeamento
    # ============================================================

    def _resolve_entity_remap(
        self,
        entity_remap,
    ):
        resolved = {}

        for source_entity_id in (
            entity_remap
        ):
            current = source_entity_id
            visited = set()

            while (
                current
                in entity_remap
            ):
                if current in visited:
                    break

                visited.add(
                    current
                )

                current = (
                    entity_remap[
                        current
                    ]
                )

            resolved[
                source_entity_id
            ] = current

        return resolved

    def _apply_remap_to_frames(
        self,
        processed_frames,
        entity_remap,
    ):
        if not entity_remap:
            return

        for frame in processed_frames:
            for person in frame.get(
                "people",
                [],
            ):
                original_entity_id = (
                    person.get(
                        "entity_id"
                    )
                )

                if (
                    original_entity_id
                    not in entity_remap
                ):
                    continue

                final_entity_id = (
                    entity_remap[
                        original_entity_id
                    ]
                )

                person[
                    "pre_reconciliation_entity_id"
                ] = original_entity_id

                person[
                    "entity_id"
                ] = final_entity_id

                person[
                    "reconciled"
                ] = True

    def _serialize_details(
        self,
        details,
    ):
        result = {}

        for key, value in (
            details.items()
        ):
            if value is None:
                result[key] = None

            else:
                result[key] = round(
                    float(value),
                    4,
                )

        return result