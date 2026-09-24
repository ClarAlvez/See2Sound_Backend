from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


BoundingBox = Tuple[
    float,
    float,
    float,
    float,
]


def cosine_similarity(
    embedding_a: Optional[torch.Tensor],
    embedding_b: Optional[torch.Tensor],
):
    if embedding_a is None or embedding_b is None:
        return None

    embedding_a = (
        embedding_a
        .float()
        .flatten()
    )

    embedding_b = (
        embedding_b
        .float()
        .flatten()
    )

    if (
        embedding_a.numel() == 0
        or embedding_b.numel() == 0
    ):
        return None

    score = F.cosine_similarity(
        embedding_a.unsqueeze(0),
        embedding_b.unsqueeze(0),
    ).item()

    return max(
        0.0,
        min(
            1.0,
            float(score),
        ),
    )

def bbox_center_similarity(
    box_a: BoundingBox,
    box_b: BoundingBox,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    center_a_x = (
        ax1 + ax2
    ) / 2.0

    center_a_y = (
        ay1 + ay2
    ) / 2.0

    center_b_x = (
        bx1 + bx2
    ) / 2.0

    center_b_y = (
        by1 + by2
    ) / 2.0

    distance_x = (
        center_a_x
        - center_b_x
    )

    distance_y = (
        center_a_y
        - center_b_y
    )

    distance = (
        distance_x ** 2
        + distance_y ** 2
    ) ** 0.5

    width_a = max(
        1.0,
        ax2 - ax1,
    )

    height_a = max(
        1.0,
        ay2 - ay1,
    )

    width_b = max(
        1.0,
        bx2 - bx1,
    )

    height_b = max(
        1.0,
        by2 - by1,
    )

    reference_size = (
        (
            width_a
            + height_a
            + width_b
            + height_b
        )
        / 4.0
    )

    normalized_distance = (
        distance
        / max(
            reference_size,
            1.0,
        )
    )

    similarity = (
        1.0
        - min(
            1.0,
            normalized_distance,
        )
    )

    return similarity

def embedding_gallery_similarity(
    embedding: Optional[torch.Tensor],
    embedding_history,
    top_k=3,
):
    if embedding is None:
        return None

    if not embedding_history:
        return None

    scores = []

    for previous_embedding in embedding_history:
        score = cosine_similarity(
            embedding,
            previous_embedding,
        )

        if score is not None:
            scores.append(
                score
            )

    if not scores:
        return None

    scores.sort(
        reverse=True
    )

    selected_scores = scores[
        :min(
            top_k,
            len(scores),
        )
    ]

    return (
        sum(selected_scores)
        / len(selected_scores)
    )


def cross_gallery_similarity(
    history_a,
    history_b,
    top_k=3,
):
    if not history_a or not history_b:
        return None

    scores = []

    for embedding_a in history_a:
        for embedding_b in history_b:
            score = cosine_similarity(
                embedding_a,
                embedding_b,
            )

            if score is not None:
                scores.append(
                    score
                )

    if not scores:
        return None

    scores.sort(
        reverse=True
    )

    selected_scores = scores[
        :min(
            top_k,
            len(scores),
        )
    ]

    return (
        sum(selected_scores)
        / len(selected_scores)
    )


def bbox_iou(
    box_a: BoundingBox,
    box_b: BoundingBox,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(
        ax1,
        bx1,
    )

    intersection_y1 = max(
        ay1,
        by1,
    )

    intersection_x2 = min(
        ax2,
        bx2,
    )

    intersection_y2 = min(
        ay2,
        by2,
    )

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        / union_area
    )


def attribute_similarity(
    attributes_a: Dict[str, float],
    attributes_b: Dict[str, float],
    allowed_attributes: Sequence[str],
    evidence_threshold=0.50,
):
    weighted_score = 0.0
    total_weight = 0.0

    for attribute in allowed_attributes:
        if (
            attribute not in attributes_a
            or attribute not in attributes_b
        ):
            continue

        score_a = max(
            0.0,
            min(
                1.0,
                float(
                    attributes_a[attribute]
                ),
            ),
        )

        score_b = max(
            0.0,
            min(
                1.0,
                float(
                    attributes_b[attribute]
                ),
            ),
        )

        evidence = max(
            score_a,
            score_b,
        )

        # Duas probabilidades muito baixas não são
        # evidência forte de identidade.
        if evidence < evidence_threshold:
            continue

        similarity = (
            1.0
            - abs(
                score_a
                - score_b
            )
        )

        weighted_score += (
            similarity
            * evidence
        )

        total_weight += evidence

    if total_weight <= 0:
        return None

    return (
        weighted_score
        / total_weight
    )


def temporal_similarity(
    current_timestamp,
    last_seen,
    window_seconds,
    floor=0.0,
):
    if current_timestamp is None or last_seen is None:
        return None

    if window_seconds <= 0:
        return None

    delta = max(
        0.0,
        float(current_timestamp)
        - float(last_seen),
    )

    score = (
        1.0
        - min(
            1.0,
            delta / window_seconds,
        )
    )

    return max(
        float(floor),
        score,
    )


def interval_gap_seconds(
    first_a,
    last_a,
    first_b,
    last_b,
):
    first_a = float(first_a)
    last_a = float(last_a)

    first_b = float(first_b)
    last_b = float(last_b)

    if (
        first_a <= last_b
        and first_b <= last_a
    ):
        return 0.0

    if last_a < first_b:
        return (
            first_b
            - last_a
        )

    return (
        first_a
        - last_b
    )


def temporal_interval_similarity(
    first_a,
    last_a,
    first_b,
    last_b,
    window_seconds,
    floor=0.0,
):
    if window_seconds <= 0:
        return None

    gap = interval_gap_seconds(
        first_a=first_a,
        last_a=last_a,
        first_b=first_b,
        last_b=last_b,
    )

    score = (
        1.0
        - min(
            1.0,
            gap / window_seconds,
        )
    )

    return max(
        float(floor),
        score,
    )


def combine_scores(
    values,
):
    total_score = 0.0
    total_weight = 0.0

    for score, weight in values:
        if score is None:
            continue

        total_score += (
            float(score)
            * float(weight)
        )

        total_weight += float(
            weight
        )

    if total_weight <= 0:
        return 0.0

    return (
        total_score
        / total_weight
    )