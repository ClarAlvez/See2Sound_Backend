
from copy import deepcopy


def apply_action_correction(
    scene,
    comparison,
    enabled=False,
    experimental_action_score=0.65,
):
    proposal = comparison.get(
        "proposal", {}
    )

    if (
        not enabled
        or not proposal.get("eligible")
    ):
        return scene

    selected_action = proposal.get(
        "promote_action"
    ) or proposal.get(
        "selected_action"
    )

    if not selected_action:
        return scene

    if not (
        0.0 <= experimental_action_score <= 1.0
    ):
        raise ValueError(
            "Score operacional inválido."
        )

    corrected = deepcopy(
        scene
    )

    original = {
        "labels": deepcopy(
            scene.get("labels", [])
        ),
        "confidence": deepcopy(
            scene.get("confidence", {})
        ),
        "temporal_action": deepcopy(
            scene.get("temporal_action", {})
        ),
    }

    add_labels = set(
        proposal.get(
            "add_labels", []
        )
    )

    remove_labels = set(
        proposal.get(
            "remove_labels", []
        )
    )

    # Somente labels de ação.
    # Nunca remover Person, Object,
    # Scene ou Atmosphere por este módulo.

    labels = [
        label
        for label in corrected.get(
            "labels", []
        )
        if label not in remove_labels
    ]

    for label in sorted(add_labels):
        if label not in labels:
            labels.append(label)

    corrected["labels"] = labels

    # Confidence original preservada.
    confidence = dict(
        corrected.get(
            "confidence", {}
        )
    )

    for label in remove_labels:
        confidence.pop(
            f"action.{label}",
            None,
        )

    for label in add_labels:
        confidence[
            f"action.{label}"
        ] = experimental_action_score

    corrected["confidence"] = (
        confidence
    )

    # Atualizar temporal_action.
    temporal = deepcopy(
        corrected.get(
            "temporal_action", {}
        )
    )

    predictions = [
        item
        for item in temporal.get(
            "predictions", []
        )
        if item.get("label")
        not in remove_labels
    ]

    present = {
        item.get("label")
        for item in predictions
    }

    for label in sorted(add_labels):
        if label not in present:
            predictions.append({
                "label": label,
                "score": experimental_action_score,
                "source": (
                    "xclip_temporal_consensus"
                ),
                "calibrated": False,
            })

    temporal["predictions"] = (
        predictions
    )

    corrected["temporal_action"] = (
        temporal
    )

    context = deepcopy(
        corrected.get("context", {})
    )

    tasks = context.setdefault(
        "spectra_tasks", {}
    )

    tasks["action"] = (
        predictions
    )

    merged_scores = (
        context.setdefault(
            "merged_label_scores", {}
        )
    )

    for label in remove_labels:
        merged_scores.pop(
            label, None
        )

    for label in add_labels:
        merged_scores[label] = (
            experimental_action_score
        )

    # A prioridade é independente
    # do score original da Spectra.
    if selected_action in labels:
        context["preferred_action"] = (
            selected_action
        )

        context[
            "preferred_action_source"
        ] = (
            "xclip_temporal_consensus"
        )

        context[
            "preferred_action_evidence"
        ] = deepcopy(
            proposal.get("evidence", {})
        )

    # Ação que envolve um veículo.
    # Pode ser usada para construir
    # uma frase sem motorista visível.
    if selected_action == "driving":
        context["vehicle_action_hint"] = {
            "action": "driving",
            "source": "xclip",
            "requires_visible_driver": False,
        }

    corrected["context"] = context

    corrected["action_original"] = (
        original
    )

    corrected["action_correction"] = {
        "applied": True,
        "selected_action": (
            selected_action
        ),
        "added": sorted(add_labels),
        "removed": sorted(remove_labels),
        "priority_only": proposal.get(
            "priority_only", False
        ),
        "evidence": deepcopy(
            proposal.get("evidence", {})
        ),
        "mode": "experimental_fusion_v2",
        "score_calibrated": False,
        "person_object_labels_removed": [],
    }

    print(
        "[FUSION V2] Ação prioritária:",
        selected_action,
    )

    print(
        "[FUSION V2] Labels adicionadas:",
        sorted(add_labels),
    )

    print(
        "[FUSION V2] Labels removidas:",
        sorted(remove_labels),
    )

    return corrected
