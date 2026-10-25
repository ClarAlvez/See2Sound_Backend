
from collections import defaultdict
from statistics import mean, median


MOTION_LABELS = {
    "walking", "running", "standing",
    "sitting", "lying_down", "crouching",
    "jumping", "dancing", "cycling",
    "climbing", "swimming", "falling",
    "still", "moving", "fast_motion",
}

ACTIVITY_LABELS = {
    "driving", "phone_use", "talking",
    "reading", "writing", "eating",
    "drinking", "computer_use",
    "instrument_playing", "cooking",
    "cleaning", "carrying", "reaching",
    "throwing", "exercising", "working",
    "playing",
}

VEHICLES = {
    "car", "truck", "bus",
    "motorcycle", "bicycle",
}

POLICY = {
    "motion": {
        "min_clips": 4,
        "full_ratio": 0.75,
        "person_ratio": 0.50,
        "median_margin": 0.04,
        "median_score": 0.0,
    },
    "activity": {
        "min_clips": 4,
        "full_ratio": 0.75,
        "person_ratio": 0.50,
        "median_margin": 0.20,
        "median_score": 0.50,
    },
}

# Objetos são evidências auxiliares.
# Não são provas de que a ação ocorre.
ACTION_CONTEXT = {
    "cooking": {
        "objects": {
            "food", "knife", "bowl",
            "pan", "pot", "stove",
            "oven", "spoon",
        },
        "min_objects": 2,
        "min_object_score": 0.35,
    },
}

PERSON_CONTEXT_POLICY = {
    "min_clips": 4,
    "min_top1_ratio": 0.33,
    "min_top2_ratio": 0.75,
    "min_full_top5_ratio": 0.50,
    "min_person_median_score": 0.14,
    "min_context_objects": 2,
}

CONTRADICTIONS = {
    "walking": {"still"},
    "running": {"still"},
    "cycling": {"still"},
    "swimming": {"still"},
    "jumping": {"still"},
}

MAX_REMOVABLE_SPECTRA_SCORE = 0.75

SPECIFICITY = {
    "cooking": 4,
    "cycling": 3,
    "driving": 3,
    "running": 3,
    "swimming": 3,
    "climbing": 3,
    "jumping": 3,
    "dancing": 3,
    "walking": 2,
    "sitting": 2,
    "standing": 1,
    "still": 1,
    "moving": 1,
    "working": 1,
    "computer_use": 2,
}


def get_spectra_actions(scene):
    predictions = (
        scene.get("temporal_action", {})
        .get("predictions", [])
    )

    if not predictions:
        predictions = (
            scene.get("context", {})
            .get("spectra_tasks", {})
            .get("action", [])
        )

    if not predictions:
        predictions = [
            {
                "label": key.split(".", 1)[1],
                "score": score,
            }
            for key, score in scene.get(
                "confidence", {}
            ).items()
            if key.startswith("action.")
        ]

    return [
        {
            "label": item["label"],
            "score": float(
                item.get("score", 0.0)
            ),
        }
        for item in predictions
        if item.get("label")
    ]


def _ranked_predictions(view, group):
    predictions = (
        view.get("groups", {})
        .get(group, [])
    )

    return sorted(
        predictions,
        key=lambda item: float(
            item.get("relative_score", 0.0)
        ),
        reverse=True,
    )


def _object_evidence(scene, action):
    config = ACTION_CONTEXT.get(
        action, {}
    )

    wanted = config.get(
        "objects", set()
    )

    confidence = scene.get(
        "confidence", {}
    )

    labels = set(
        scene.get("labels", [])
    )

    objects = []

    for label in sorted(wanted):
        score = float(
            confidence.get(
                f"object.{label}", 0.0
            )
        )

        if (
            label in labels
            and score >= config.get(
                "min_object_score", 0.35
            )
        ):
            objects.append({
                "label": label,
                "score": score,
            })

    return objects


def _scene_vehicle_evidence(scene):
    labels = set(
        scene.get("labels", [])
    )

    confidence = scene.get(
        "confidence", {}
    )

    return [
        {
            "label": label,
            "score": float(
                confidence.get(
                    f"object.{label}", 0.0
                )
            ),
        }
        for label in sorted(VEHICLES)
        if (
            label in labels
            and float(
                confidence.get(
                    f"object.{label}", 0.0
                )
            ) >= 0.70
        )
    ]


def summarize_xclip(clips):
    stats = defaultdict(
        lambda: {
            "full_top1": set(),
            "full_top5": set(),
            "person_top1": set(),
            "person_top2": set(),
            "full_scores": [],
            "person_scores": [],
            "full_margins": [],
        }
    )

    # Apenas janelas com evidências X-CLIP.
    total_clips = len(clips)

    # O índice local da pessoa não representa
    # identidade persistente ao longo do vídeo.
    # Guardamos apenas se ao menos uma pessoa
    # apresentou evidência em cada janela.
    person_counts = []

    for index, clip in enumerate(clips):
        person_views = [
            view
            for view in clip.get("views", [])
            if view.get("view") == "person"
        ]

        person_counts.append(
            len(person_views)
        )

        for view in clip.get("views", []):
            view_type = view.get("view")

            if view_type not in {
                "full_frame", "person"
            }:
                continue

            for group in (
                "motion", "activity"
            ):
                predictions = (
                    _ranked_predictions(
                        view, group
                    )
                )

                if not predictions:
                    continue

                top_score = float(
                    predictions[0].get(
                        "relative_score", 0.0
                    )
                )

                second_score = (
                    float(
                        predictions[1].get(
                            "relative_score", 0.0
                        )
                    )
                    if len(predictions) > 1
                    else 0.0
                )

                for rank, prediction in enumerate(
                    predictions,
                    start=1,
                ):
                    label = prediction.get(
                        "label"
                    )

                    if not label:
                        continue

                    score = float(
                        prediction.get(
                            "relative_score", 0.0
                        )
                    )

                    item = stats[
                        (group, label)
                    ]

                    if view_type == "full_frame":
                        if rank == 1:
                            item[
                                "full_top1"
                            ].add(index)

                            item[
                                "full_margins"
                            ].append(
                                top_score
                                - second_score
                            )

                        if rank <= 5:
                            item[
                                "full_top5"
                            ].add(index)

                        item[
                            "full_scores"
                        ].append(score)

                    else:
                        if rank == 1:
                            item[
                                "person_top1"
                            ].add(index)

                        if rank <= 2:
                            item[
                                "person_top2"
                            ].add(index)

                        item[
                            "person_scores"
                        ].append(score)

    results = []

    for (group, label), item in stats.items():
        def ratio(name):
            if total_clips == 0:
                return 0.0

            return (
                len(item[name])
                / total_clips
            )

        def middle(values):
            return (
                median(values)
                if values else 0.0
            )

        results.append({
            "group": group,
            "label": label,
            "total_clips": total_clips,
            "full_frame_first_count": len(
                item["full_top1"]
            ),
            "person_supported_clips": len(
                item["person_top1"]
            ),
            "full_frame_ratio": round(
                ratio("full_top1"), 4
            ),
            "full_top5_ratio": round(
                ratio("full_top5"), 4
            ),
            "person_frame_ratio": round(
                ratio("person_top1"), 4
            ),
            "person_top2_ratio": round(
                ratio("person_top2"), 4
            ),
            "median_relative_score": round(
                middle(
                    item["full_scores"]
                ), 5
            ),
            "median_person_score": round(
                middle(
                    item["person_scores"]
                ), 5
            ),
            "median_margin": round(
                middle(
                    item["full_margins"]
                ), 5
            ),
            "person_view_max": max(
                person_counts,
                default=0,
            ),
            "evidence_passed": False,
            "evidence_route": None,
            "supporting_objects": [],
        })

    return results


def _full_frame_route(item, scene):
    policy = POLICY[item["group"]]

    if (
        item["total_clips"]
        < policy["min_clips"]
    ):
        return False

    if (
        item["full_frame_ratio"]
        < policy["full_ratio"]
    ):
        return False

    if (
        item["median_margin"]
        < policy["median_margin"]
    ):
        return False

    if (
        item["median_relative_score"]
        < policy["median_score"]
    ):
        return False

    # Driving pode ser reconhecido
    # sem um motorista visível, mas exige
    # um veículo corroborando a previsão.
    if item["label"] == "driving":
        return bool(
            _scene_vehicle_evidence(scene)
        )

    return (
        item["person_frame_ratio"]
        >= policy["person_ratio"]
    )


def _person_context_route(
    item,
    scene,
):
    label = item["label"]

    if label not in ACTION_CONTEXT:
        return False

    p = PERSON_CONTEXT_POLICY

    if (
        item["total_clips"]
        < p["min_clips"]
    ):
        return False

    # Na v3 contextual, exigimos uma única
    # pessoa por janela para não confundir
    # ações entre pessoas diferentes.
    # Isso ainda não equivale a tracking.
    if item["person_view_max"] != 1:
        return False

    if (
        item["person_frame_ratio"]
        < p["min_top1_ratio"]
    ):
        return False

    if (
        item["person_top2_ratio"]
        < p["min_top2_ratio"]
    ):
        return False

    if (
        item["full_top5_ratio"]
        < p["min_full_top5_ratio"]
    ):
        return False

    if (
        item["median_person_score"]
        < p["min_person_median_score"]
    ):
        return False

    objects = _object_evidence(
        scene, label
    )

    item["supporting_objects"] = objects

    required = max(
        p["min_context_objects"],
        ACTION_CONTEXT[label].get(
            "min_objects", 2
        ),
    )

    return len(objects) >= required


def _evaluate_evidence(
    item,
    scene,
):
    if _person_context_route(
        item, scene
    ):
        item["evidence_passed"] = True
        item["evidence_route"] = (
            "person_context"
        )
        return

    if _full_frame_route(
        item, scene
    ):
        item["evidence_passed"] = True
        item["evidence_route"] = (
            "full_frame"
        )


def _choose_candidate(evidence):
    valid = [
        item
        for item in evidence
        if item["evidence_passed"]
    ]

    if not valid:
        return None

    def ranking(item):
        return (
            SPECIFICITY.get(
                item["label"], 2
            ),
            (
                item["person_top2_ratio"]
                if item["evidence_route"]
                == "person_context"
                else item["full_frame_ratio"]
            ),
            item["median_margin"],
        )

    valid.sort(
        key=ranking,
        reverse=True,
    )

    # Se ações concorrentes de igual
    # especificidade foram aprovadas,
    # não escolher arbitrariamente.
    if len(valid) > 1:
        first = valid[0]
        second = valid[1]

        if (
            first["group"] == second["group"]
            and SPECIFICITY.get(
                first["label"], 2
            ) == SPECIFICITY.get(
                second["label"], 2
            )
        ):
            return None

    return valid[0]


def _action_score(scene, label):
    return float(
        scene.get("confidence", {})
        .get(
            f"action.{label}", 0.0
        )
    )


def compare_scene(scene, clips):
    spectra = get_spectra_actions(
        scene
    )

    labels = set(
        scene.get("labels", [])
    )

    evidence = summarize_xclip(
        clips
    )

    for item in evidence:
        _evaluate_evidence(
            item, scene
        )

    selected = _choose_candidate(
        evidence
    )

    add_labels = []
    remove_labels = []
    priority = None

    if selected:
        priority = selected["label"]

        if priority not in labels:
            add_labels.append(
                priority
            )

        contradictions = (
            CONTRADICTIONS.get(
                priority, set()
            )
        )

        for label in sorted(
            contradictions
        ):
            if (
                label in labels
                and _action_score(
                    scene, label
                )
                < MAX_REMOVABLE_SPECTRA_SCORE
            ):
                remove_labels.append(
                    label
                )

    has_changes = bool(
        add_labels or remove_labels
    )

    if not clips:
        status = "no_clips"

    elif selected is None:
        status = "inconclusive_xclip"

    elif has_changes:
        status = "correction_proposed"

    else:
        status = "priority_proposed"

    proposal = {
        "eligible": selected is not None,
        "selected_action": priority,
        "promote_action": priority,
        "add_labels": sorted(
            add_labels
        ),
        "remove_labels": sorted(
            remove_labels
        ),
        "priority_only": (
            selected is not None
            and not has_changes
        ),
        "has_label_changes": has_changes,
        "evidence": selected,
        "reason": (
            "temporal_consensus"
            if selected else
            "insufficient_or_ambiguous_evidence"
        ),
    }

    return {
        "status": status,
        "spectra_predictions": spectra,
        "xclip_predictions": evidence,
        "proposal": proposal,
        "correction_applied": False,
    }


def build_summary(scenes):
    counts = defaultdict(int)
    total_clips = 0
    total_ms = 0.0
    inference_times = []

    for scene in scenes:
        verification = scene.get(
            "action_verification", {}
        )

        comparison = verification.get(
            "comparison", {}
        )

        counts[
            comparison.get(
                "status", "no_clips"
            )
        ] += 1

        for clip in verification.get(
            "xclip_clips", []
        ):
            total_clips += 1

            total_ms += float(
                clip.get(
                    "total_ms", 0.0
                )
            )

            for view in clip.get(
                "views", []
            ):
                inference_times.append(
                    float(
                        view.get(
                            "inference_ms", 0.0
                        )
                    )
                )

    return {
        "model": (
            "microsoft/"
            "xclip-base-patch32-16-frames"
        ),
        "total_scenes": len(scenes),
        "total_clips": total_clips,
        "correction_proposed": counts[
            "correction_proposed"
        ],
        "priority_proposed": counts[
            "priority_proposed"
        ],
        "inconclusive_xclip": counts[
            "inconclusive_xclip"
        ],
        "no_clips": counts[
            "no_clips"
        ],
        "mean_inference_ms": round(
            mean(inference_times), 2
        ) if inference_times else 0.0,
        "total_clip_processing_ms": round(
            total_ms, 2
        ),
        "corrections_applied": sum(
            bool(
                scene.get(
                    "action_correction", {}
                ).get("applied", False)
            )
            for scene in scenes
        ),
    }
