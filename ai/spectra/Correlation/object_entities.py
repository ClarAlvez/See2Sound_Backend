from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


BoundingBox = Tuple[
    float,
    float,
    float,
    float,
]


@dataclass
class ObjectDetection:
    detection_id: str

    frame_id: int
    timestamp: float

    label: str
    label_score: float

    bbox: BoundingBox

    crop_path: Optional[str] = None

    detector_confidence: Optional[float] = None

    scene_id: Optional[str] = None


@dataclass
class ActiveObjectTrack:
    track_id: str
    object_id: str

    label: str

    frame_id: int
    timestamp: float

    bbox: BoundingBox

    label_score: float

    scene_id: Optional[str] = None


@dataclass
class ObjectEntity:
    object_id: str
    label: str

    first_seen: float
    last_seen: float

    first_frame: int
    last_frame: int

    observation_count: int = 0

    score_sum: float = 0.0
    max_score: float = 0.0

    last_bbox: Optional[BoundingBox] = None

    track_ids: List[str] = field(
        default_factory=list
    )

    scenes_seen: Dict[str, int] = field(
        default_factory=dict
    )

    example_crops: List[str] = field(
        default_factory=list
    )

    observation_frames: List[int] = field(
        default_factory=list
    )

    def to_dict(self):
        average_score = 0.0

        if self.observation_count > 0:
            average_score = (
                self.score_sum
                / self.observation_count
            )

        return {
            "object_id": self.object_id,
            "label": self.label,

            "track_ids": list(
                self.track_ids
            ),

            "track_count": len(
                self.track_ids
            ),

            "first_seen": round(
                self.first_seen,
                4,
            ),

            "last_seen": round(
                self.last_seen,
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

            "average_score": round(
                average_score,
                4,
            ),

            "max_score": round(
                self.max_score,
                4,
            ),

            "last_bbox": (
                list(self.last_bbox)
                if self.last_bbox
                is not None
                else None
            ),

            "scenes_seen": (
                self.scenes_seen
            ),

            "example_crops": (
                self.example_crops
            ),

            "observation_frames": (
                self.observation_frames
            ),
        }