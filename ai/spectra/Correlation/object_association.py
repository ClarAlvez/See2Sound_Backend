from typing import Dict, List

from ai.spectra.Correlation.similarity import (
    bbox_center_similarity,
)


def intersection_area(
    box_a,
    box_b,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    width = max(
        0.0,
        x2 - x1,
    )

    height = max(
        0.0,
        y2 - y1,
    )

    return (
        width
        * height
    )


def bbox_area(box):
    x1, y1, x2, y2 = box

    return max(
        0.0,
        x2 - x1,
    ) * max(
        0.0,
        y2 - y1,
    )


def object_containment_ratio(
    object_box,
    person_box,
):
    object_area = bbox_area(
        object_box
    )

    if object_area <= 0:
        return 0.0

    return (
        intersection_area(
            object_box,
            person_box,
        )
        / object_area
    )


def object_center_inside_person(
    object_box,
    person_box,
):
    ox1, oy1, ox2, oy2 = (
        object_box
    )

    px1, py1, px2, py2 = (
        person_box
    )

    center_x = (
        ox1 + ox2
    ) / 2.0

    center_y = (
        oy1 + oy2
    ) / 2.0

    return (
        px1 <= center_x <= px2
        and py1 <= center_y <= py2
    )


class PersonObjectAssociator:
    def __init__(
        self,
        config,
    ):
        self.config = config

    def associate(
        self,
        people: List[Dict],
        objects: List[Dict],
    ):
        associations = []

        for object_data in objects:
            candidates = []

            for person in people:
                candidate = (
                    self._score_pair(
                        person,
                        object_data,
                    )
                )

                if candidate is not None:
                    candidates.append(
                        candidate
                    )

            candidates.sort(
                key=lambda item: item[
                    "score"
                ],
                reverse=True,
            )

            if not candidates:
                continue

            best = candidates[0]

            if (
                best["score"]
                < self.config.object_association_min_score
            ):
                continue

            if len(candidates) > 1:
                margin = (
                    best["score"]
                    - candidates[1][
                        "score"
                    ]
                )

                if (
                    margin
                    < self.config.object_association_min_margin
                ):
                    continue

            associations.append(
                best
            )

        return associations

    def _score_pair(
        self,
        person,
        object_data,
    ):
        person_box = tuple(
            person["bbox"]
        )

        object_box = tuple(
            object_data["bbox"]
        )

        containment = (
            object_containment_ratio(
                object_box,
                person_box,
            )
        )

        center_score = (
            bbox_center_similarity(
                object_box,
                person_box,
            )
        )

        center_inside = (
            object_center_inside_person(
                object_box,
                person_box,
            )
        )

        score = (
            containment * 0.70
            + center_score * 0.30
        )

        if center_inside:
            score = max(
                score,
                0.65,
            )

        score = min(
            score,
            1.0,
        )

        relation = (
            self._infer_relation(
                label=object_data[
                    "label"
                ],

                containment=(
                    containment
                ),

                center_inside=(
                    center_inside
                ),
            )
        )

        if relation is None:
            return None

        return {
            "entity_id": person[
                "entity_id"
            ],

            "person_track_id": (
                person["track_id"]
            ),

            "object_id": object_data[
                "object_id"
            ],

            "object_track_id": (
                object_data[
                    "track_id"
                ]
            ),

            "object_label": (
                object_data["label"]
            ),

            "relation": relation,

            "score": round(
                score,
                4,
            ),
        }

    def _infer_relation(
        self,
        label,
        containment,
        center_inside,
    ):
        if (
            label
            in self.config.object_wearable_labels
            and center_inside
        ):
            return "wears"

        if (
            label
            in self.config.object_carried_labels
            and (
                center_inside
                or containment >= 0.35
            )
        ):
            return "carries"

        if (
            label
            in self.config.object_held_labels
            and (
                center_inside
                or containment >= 0.35
            )
        ):
            return "holds"

        if (
            center_inside
            or containment >= 0.20
        ):
            return "near"

        return None

class ObjectAssociationMemory:
    def __init__(
        self,
        config,
    ):
        self.config = config

        self.relations = {}

    def register(
        self,
        association,
        frame_id,
        timestamp,
    ):
        entity_id = association[
            "entity_id"
        ]

        object_label = association[
            "object_label"
        ]

        relation = association[
            "relation"
        ]

        key = (
            entity_id,
            object_label,
            relation,
        )

        if key not in self.relations:
            self.relations[key] = {
                "entity_id": entity_id,

                "object_label": (
                    object_label
                ),

                "relation": relation,

                "observation_count": 0,

                "best_score": 0.0,

                "first_seen": timestamp,

                "last_seen": timestamp,

                "frame_ids": [],

                "object_ids": [],
            }

        record = self.relations[
            key
        ]

        record[
            "observation_count"
        ] += 1

        record["best_score"] = max(
            record["best_score"],
            association["score"],
        )

        record["last_seen"] = max(
            record["last_seen"],
            timestamp,
        )

        if (
            frame_id
            not in record["frame_ids"]
        ):
            record[
                "frame_ids"
            ].append(
                frame_id
            )

        object_id = association[
            "object_id"
        ]

        if (
            object_id
            not in record["object_ids"]
        ):
            record[
                "object_ids"
            ].append(
                object_id
            )

    def remap_entities(
        self,
        entity_remap,
    ):
        if not entity_remap:
            return

        old_relations = list(
            self.relations.values()
        )

        self.relations = {}

        for record in old_relations:
            entity_id = record[
                "entity_id"
            ]

            while (
                entity_id
                in entity_remap
            ):
                entity_id = (
                    entity_remap[
                        entity_id
                    ]
                )

            record[
                "entity_id"
            ] = entity_id

            key = (
                entity_id,
                record[
                    "object_label"
                ],
                record[
                    "relation"
                ],
            )

            if key not in self.relations:
                self.relations[
                    key
                ] = record

            else:
                target = self.relations[
                    key
                ]

                target[
                    "observation_count"
                ] += record[
                    "observation_count"
                ]

                target[
                    "best_score"
                ] = max(
                    target[
                        "best_score"
                    ],
                    record[
                        "best_score"
                    ],
                )

                target[
                    "last_seen"
                ] = max(
                    target[
                        "last_seen"
                    ],
                    record[
                        "last_seen"
                    ],
                )

                target[
                    "frame_ids"
                ] = sorted(
                    set(
                        target[
                            "frame_ids"
                        ]
                        + record[
                            "frame_ids"
                        ]
                    )
                )

                target[
                    "object_ids"
                ] = sorted(
                    set(
                        target[
                            "object_ids"
                        ]
                        + record[
                            "object_ids"
                        ]
                    )
                )

    def to_dict(self):
        grouped = {}

        for record in (
            self.relations.values()
        ):
            entity_id = record[
                "entity_id"
            ]

            if entity_id not in grouped:
                grouped[
                    entity_id
                ] = []

            grouped[
                entity_id
            ].append(
                {
                    **record,

                    "best_score": round(
                        record[
                            "best_score"
                        ],
                        4,
                    ),

                    "first_seen": round(
                        record[
                            "first_seen"
                        ],
                        4,
                    ),

                    "last_seen": round(
                        record[
                            "last_seen"
                        ],
                        4,
                    ),

                    "persistent": (
                        record[
                            "observation_count"
                        ]
                        >= self.config.object_relation_min_observations
                    ),
                }
            )

        return grouped