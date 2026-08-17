from collections import Counter
from typing import Dict, List, Any

def deduplicate_predictions(
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    best_by_label = {}

    for prediction in predictions:
        label = prediction.get("label")

        if not label:
            continue

        score = float(prediction.get("score", 0.0))

        current = best_by_label.get(label)

        if current is None or score > float(current.get("score", 0.0)):
            best_by_label[label] = prediction
            best_by_label[label]["score"] = round(score, 4)

    deduplicated = list(best_by_label.values())

    deduplicated.sort(
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )

    return deduplicated

def postprocess_temporal_actions(frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recebe os resultados de Actions de vários frames de uma mesma cena/trecho
    e gera labels temporais mais estáveis.
    """

    label_scores = {}
    label_counts = Counter()

    total_frames = len(frame_results)

    for frame_result in frame_results:
        predictions = frame_result.get("predictions", [])

        for prediction in predictions:
            label = prediction.get("label")
            score = float(prediction.get("score", 0))

            if not label:
                continue

            label_counts[label] += 1
            label_scores[label] = max(label_scores.get(label, 0), score)

    final_predictions = []

    for label, score in label_scores.items():
        final_predictions.append(
            {
                "label": label,
                "score": round(score, 4),
                "source": "frame_action",
                "frame_count": label_counts[label],
            }
        )

    has_fast_motion = label_counts["fast_motion"] >= 2
    has_moving = label_counts["moving"] >= 1
    has_exercising = label_counts["exercising"] >= 1

    # Regra para corrida em vídeo.
    if has_fast_motion and has_moving:
        final_predictions.append(
            {
                "label": "running",
                "score": 0.7,
                "source": "temporal_rule",
                "frame_count": label_counts["fast_motion"] + label_counts["moving"],
            }
        )

    elif has_fast_motion and total_frames >= 3:
        final_predictions.append(
            {
                "label": "running",
                "score": 0.65,
                "source": "temporal_rule",
                "frame_count": label_counts["fast_motion"],
            }
        )

    elif has_moving and has_exercising:
        final_predictions.append(
            {
                "label": "running",
                "score": 0.6,
                "source": "temporal_rule",
                "frame_count": label_counts["moving"] + label_counts["exercising"],
            }
        )

    # Se running apareceu por regra temporal, jumping isolado costuma ser ruído de passada.
    has_running = any(item["label"] == "running" for item in final_predictions)

    if has_running and label_counts["jumping"] <= 1:
        final_predictions = [
            item
            for item in final_predictions
            if item["label"] != "jumping"
        ]

    # Se existe movimento rápido, standing fraco/repetido pouco não ajuda.
    if has_fast_motion:
        final_predictions = [
            item
            for item in final_predictions
            if item["label"] != "standing"
        ]

    if has_running:
        final_predictions = [
            item
            for item in final_predictions
            if item["label"] not in ["standing", "still"]
        ]

    final_predictions = deduplicate_predictions(final_predictions)

    return {
        "task_name": "action_temporal",
        "source": "temporal_postprocess",
        "frames_analyzed": total_frames,
        "predictions": final_predictions,
    }