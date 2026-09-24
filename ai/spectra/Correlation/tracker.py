from typing import Dict, List, Tuple

from ai.spectra.Correlation.config import (
    CorrelationConfig,
)
from ai.spectra.Correlation.entities import (
    ActiveTrack,
    PersonDetection,
)
from ai.spectra.Correlation.similarity import (
    attribute_similarity,
    bbox_center_similarity,
    bbox_iou,
    combine_scores,
    cosine_similarity,
)


class PersonTracker:
    def __init__(
        self,
        config: CorrelationConfig,
    ):
        self.config = config

        self.active_tracks: Dict[
            str,
            ActiveTrack,
        ] = {}

        self.next_track_id = 1

        self.total_tracks_created = 0

    # ============================================================
    # Estado
    # ============================================================

    def clear(self):
        self.active_tracks.clear()

    def expire_old_tracks(
        self,
        timestamp,
    ):
        expired = []

        for (
            track_id,
            track,
        ) in self.active_tracks.items():

            time_difference = (
                float(timestamp)
                - float(
                    track.timestamp
                )
            )

            if (
                time_difference
                > self.config.short_term_max_seconds
            ):
                expired.append(
                    track_id
                )

        for track_id in expired:
            self.active_tracks.pop(
                track_id,
                None,
            )

    # ============================================================
    # Match entre detecção e track
    # ============================================================

    def match(
        self,
        detections: List[
            PersonDetection
        ],
        timestamp: float,
    ):
        self.expire_old_tracks(
            timestamp
        )

        candidates: List[
            Tuple[
                float,
                int,
                str,
            ]
        ] = []

        for (
            detection_index,
            detection,
        ) in enumerate(
            detections
        ):

            for (
                track_id,
                track,
            ) in self.active_tracks.items():

                embedding_score = (
                    cosine_similarity(
                        detection.embedding,
                        track.embedding,
                    )
                )

                iou_score = bbox_iou(
                    detection.bbox,
                    track.bbox,
                )

                center_score = (
                    bbox_center_similarity(
                        detection.bbox,
                        track.bbox,
                    )
                )

                attribute_score = (
                    attribute_similarity(
                        detection.attributes,
                        track.attributes,
                        self.config.stable_attributes,
                    )
                )

                spatial_score = max(
                    iou_score,
                    center_score,
                )

                # Mesmo com aparência semelhante,
                # a detecção precisa estar fisicamente
                # próxima do track anterior.
                if (
                    spatial_score
                    < self.config.short_term_min_spatial_score
                ):
                    continue

                score = combine_scores([
                    (
                        embedding_score,
                        self.config.embedding_weight,
                    ),

                    (
                        iou_score,
                        self.config.iou_weight,
                    ),

                    (
                        center_score,
                        self.config.center_weight,
                    ),

                    (
                        attribute_score,
                        self.config.attribute_weight,
                    ),
                ])

                if (
                    score
                    >= self.config.short_term_min_score
                ):
                    candidates.append(
                        (
                            score,
                            detection_index,
                            track_id,
                        )
                    )

        candidates.sort(
            key=lambda value: value[
                0
            ],
            reverse=True,
        )

        matched_detections = set()
        matched_tracks = set()

        matches = {}

        for (
            score,
            detection_index,
            track_id,
        ) in candidates:

            if (
                detection_index
                in matched_detections
            ):
                continue

            if (
                track_id
                in matched_tracks
            ):
                continue

            track = (
                self.active_tracks[
                    track_id
                ]
            )

            matched_detections.add(
                detection_index
            )

            matched_tracks.add(
                track_id
            )

            matches[
                detection_index
            ] = {
                "track_id": (
                    track_id
                ),

                "entity_id": (
                    track.entity_id
                ),

                "score": float(
                    score
                ),

                "source": (
                    "existing_track"
                ),
            }

        return matches

    # ============================================================
    # Criação de track
    # ============================================================

    def create_track(
        self,
        entity_id: str,
        detection: PersonDetection,
    ):
        track_id = (
            f"track_"
            f"{self.next_track_id:04d}"
        )

        self.next_track_id += 1
        self.total_tracks_created += 1

        self.active_tracks[
            track_id
        ] = ActiveTrack(
            track_id=track_id,

            entity_id=entity_id,

            frame_id=(
                detection.frame_id
            ),

            timestamp=(
                detection.timestamp
            ),

            bbox=(
                detection.bbox
            ),

            embedding=(
                detection.embedding
            ),

            attributes=(
                detection.attributes.copy()
            ),

            scene_id=(
                detection.scene_id
            ),
        )

        return track_id

    # ============================================================
    # Atualização de track
    # ============================================================

    def update_track(
        self,
        track_id: str,
        detection: PersonDetection,
    ):
        track = (
            self.active_tracks.get(
                track_id
            )
        )

        if track is None:
            raise KeyError(
                "Track não encontrado: "
                f"{track_id}"
            )

        track.frame_id = (
            detection.frame_id
        )

        track.timestamp = (
            detection.timestamp
        )

        track.bbox = (
            detection.bbox
        )

        track.embedding = (
            detection.embedding
        )

        track.attributes = (
            detection.attributes.copy()
        )

        track.scene_id = (
            detection.scene_id
        )

    def get_track(
        self,
        track_id,
    ):
        return (
            self.active_tracks.get(
                track_id
            )
        )