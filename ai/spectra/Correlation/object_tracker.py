from typing import Dict, List

from ai.spectra.Correlation.config import (
    CorrelationConfig,
)
from ai.spectra.Correlation.object_entities import (
    ActiveObjectTrack,
    ObjectDetection,
)
from ai.spectra.Correlation.similarity import (
    bbox_center_similarity,
    bbox_iou,
    combine_scores,
)


class ObjectTracker:
    def __init__(
        self,
        config: CorrelationConfig,
    ):
        self.config = config

        self.active_tracks: Dict[
            str,
            ActiveObjectTrack,
        ] = {}

        self.next_track_id = 1
        self.total_tracks_created = 0

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

            difference = (
                timestamp
                - track.timestamp
            )

            if (
                difference
                > self.config.object_track_max_seconds
            ):
                expired.append(
                    track_id
                )

        for track_id in expired:
            self.active_tracks.pop(
                track_id,
                None,
            )

    def match(
        self,
        detections: List[
            ObjectDetection
        ],
        timestamp: float,
    ):
        self.expire_old_tracks(
            timestamp
        )

        candidates = []

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

                if (
                    detection.label
                    != track.label
                ):
                    continue

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

                spatial_score = max(
                    iou_score,
                    center_score,
                )

                if (
                    spatial_score
                    < self.config.object_track_min_spatial_score
                ):
                    continue

                score = combine_scores([
                    (
                        iou_score,
                        self.config.object_track_iou_weight,
                    ),

                    (
                        center_score,
                        self.config.object_track_center_weight,
                    ),
                ])

                if (
                    score
                    >= self.config.object_track_min_score
                ):
                    candidates.append(
                        (
                            score,
                            detection_index,
                            track_id,
                        )
                    )

        candidates.sort(
            reverse=True,
            key=lambda item: item[0],
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

            if track_id in matched_tracks:
                continue

            track = self.active_tracks[
                track_id
            ]

            matched_detections.add(
                detection_index
            )

            matched_tracks.add(
                track_id
            )

            matches[
                detection_index
            ] = {
                "track_id": track_id,

                "object_id": (
                    track.object_id
                ),

                "score": float(
                    score
                ),
            }

        return matches

    def create_track(
        self,
        object_id,
        detection,
    ):
        track_id = (
            f"object_track_"
            f"{self.next_track_id:04d}"
        )

        self.next_track_id += 1
        self.total_tracks_created += 1

        self.active_tracks[
            track_id
        ] = ActiveObjectTrack(
            track_id=track_id,

            object_id=object_id,

            label=detection.label,

            frame_id=detection.frame_id,

            timestamp=detection.timestamp,

            bbox=detection.bbox,

            label_score=(
                detection.label_score
            ),

            scene_id=(
                detection.scene_id
            ),
        )

        return track_id

    def update_track(
        self,
        track_id,
        detection,
    ):
        track = self.active_tracks[
            track_id
        ]

        track.frame_id = (
            detection.frame_id
        )

        track.timestamp = (
            detection.timestamp
        )

        track.bbox = (
            detection.bbox
        )

        track.label_score = (
            detection.label_score
        )

        track.scene_id = (
            detection.scene_id
        )