from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


BoundingBox = Tuple[
    float,
    float,
    float,
    float,
]


@dataclass
class PersonDetection:
    detection_id: str

    frame_id: int
    timestamp: float

    bbox: BoundingBox

    crop_path: Optional[str] = None

    attributes: Dict[str, float] = field(
        default_factory=dict
    )

    embedding: Optional[torch.Tensor] = None

    scene_id: Optional[str] = None


@dataclass
class ActiveTrack:
    track_id: str
    entity_id: str

    frame_id: int
    timestamp: float

    bbox: BoundingBox

    embedding: Optional[torch.Tensor]

    attributes: Dict[str, float]

    scene_id: Optional[str]


@dataclass
class PersonEntity:
    entity_id: str

    first_seen: float
    last_seen: float

    first_frame: int
    last_frame: int

    observation_count: int = 0

    # ============================================================
    # Tracks associados à entidade
    # ============================================================

    track_ids: List[str] = field(
        default_factory=list
    )

    # ============================================================
    # Aparência
    # ============================================================

    embedding: Optional[torch.Tensor] = None

    embedding_count: int = 0

    embedding_history: List[torch.Tensor] = field(
        default_factory=list
    )

    # ============================================================
    # Atributos
    # ============================================================

    attributes: Dict[str, float] = field(
        default_factory=dict
    )

    attribute_counts: Dict[str, int] = field(
        default_factory=dict
    )

    recent_attributes: Dict[str, float] = field(
        default_factory=dict
    )

    accessory_counts: Dict[str, int] = field(
        default_factory=dict
    )

    # ============================================================
    # Contexto
    # ============================================================

    scenes_seen: Dict[str, int] = field(
        default_factory=dict
    )

    example_crops: List[str] = field(
        default_factory=list
    )

    aliases: Dict[str, float] = field(
        default_factory=dict
    )

    observation_frames: List[int] = field(
        default_factory=list
    )

    observation_timestamps: List[float] = field(
        default_factory=list
    )

    def to_dict(self):
        accessories = {}

        if self.observation_count > 0:
            for (
                label,
                count,
            ) in self.accessory_counts.items():

                accessories[label] = {
                    "observations": count,

                    "frequency": round(
                        count
                        / self.observation_count,
                        4,
                    ),
                }

        return {
            "entity_id": (
                self.entity_id
            ),

            "track_ids": list(
                self.track_ids
            ),

            "track_count": len(
                self.track_ids
            ),

            "first_seen": round(
                float(
                    self.first_seen
                ),
                4,
            ),

            "last_seen": round(
                float(
                    self.last_seen
                ),
                4,
            ),

            "first_frame": (
                self.first_frame
            ),

            "last_frame": (
                self.last_frame
            ),

            "observation_count": (
                self.observation_count
            ),

            "embedding_count": (
                self.embedding_count
            ),

            "embedding_dimension": (
                int(
                    self.embedding.shape[
                        0
                    ]
                )
                if self.embedding
                is not None
                else None
            ),

            "embedding_history_size": (
                len(
                    self.embedding_history
                )
            ),

            "attributes": {
                key: round(
                    float(value),
                    4,
                )
                for key, value
                in self.attributes.items()
            },

            "recent_attributes": {
                key: round(
                    float(value),
                    4,
                )
                for key, value
                in self.recent_attributes.items()
            },

            "accessories": (
                accessories
            ),

            "scenes_seen": (
                self.scenes_seen
            ),

            "example_crops": (
                self.example_crops
            ),

            "aliases": (
                self.aliases
            ),

            "observation_frames": (
                self.observation_frames
            ),

            "observation_timestamps": [
                round(
                    float(timestamp),
                    4,
                )
                for timestamp
                in self.observation_timestamps
            ],
        }