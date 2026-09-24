from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CorrelationConfig:
    device: Optional[str] = None

    appearance_backbone: str = "resnet18"
    pretrained: bool = True

    # ============================================================
    # Tracking local
    # ============================================================

    short_term_max_seconds: float = 2.0

    short_term_min_score: float = 0.45
    short_term_min_spatial_score: float = 0.30

    embedding_weight: float = 0.45
    iou_weight: float = 0.30
    center_weight: float = 0.20
    attribute_weight: float = 0.05

    reset_tracks_on_scene_change: bool = True

    # ============================================================
    # Atributos
    # ============================================================

    attribute_positive_threshold: float = 0.50
    attribute_evidence_threshold: float = 0.50

    recent_attribute_alpha: float = 0.35

    # ============================================================
    # Galeria visual
    # ============================================================

    max_example_crops_per_entity: int = 5

    max_embedding_history: int = 12

    gallery_top_k: int = 3

    embedding_history_duplicate_threshold: float = 0.985

    # ============================================================
    # Re-identificação entre tracks
    # ============================================================

    reidentification_min_score: float = 0.72
    reidentification_min_margin: float = 0.05

    reid_gallery_weight: float = 0.45
    reid_centroid_weight: float = 0.25

    reid_stable_attribute_weight: float = 0.15
    reid_appearance_attribute_weight: float = 0.10

    reid_temporal_weight: float = 0.05

    temporal_reid_window_seconds: float = 30.0
    temporal_reid_floor: float = 0.15

    # ============================================================
    # Reconciliação
    # ============================================================

    enable_reconciliation: bool = True

    reconcile_min_score: float = 0.78

    reconcile_gallery_top_k: int = 3

    reconcile_gallery_weight: float = 0.45
    reconcile_centroid_weight: float = 0.20

    reconcile_stable_attribute_weight: float = 0.15
    reconcile_appearance_attribute_weight: float = 0.10

    reconcile_temporal_weight: float = 0.10

    reconcile_temporal_window_seconds: float = 12.0
    reconcile_temporal_floor: float = 0.05

    # ============================================================
    # Atributos estáveis
    # ============================================================

    stable_attributes: List[str] = field(
        default_factory=lambda: [
            "blonde_hair",
            "brown_hair",
            "black_hair",
            "red_hair",
            "gray_hair",

            "short_hair",
            "long_hair",
            "curly_hair",
            "straight_hair",
        ]
    )

    # ============================================================
    # Aparência atual
    # ============================================================

    appearance_attributes: List[str] = field(
        default_factory=lambda: [
            "red_clothes",
            "blue_clothes",
            "black_clothes",
            "white_clothes",
            "green_clothes",
            "yellow_clothes",

            "glasses",
            "hat",
            "cap",
            "backpack",
            "bag",
        ]
    )

    accessory_attributes: List[str] = field(
        default_factory=lambda: [
            "glasses",
            "hat",
            "cap",
            "backpack",
            "bag",
        ]
    )