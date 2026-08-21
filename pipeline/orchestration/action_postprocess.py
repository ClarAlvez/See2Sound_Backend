from collections import Counter
from typing import Any, Dict, List


def deduplicate_predictions(
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    best_by_label: Dict[str, Dict[str, Any]] = {}

    for prediction in predictions:
        label = prediction.get("label")

        if not label:
            continue

        score = float(prediction.get("score", 0.0))
        current = best_by_label.get(label)

        if current is None or score > float(current.get("score", 0.0)):
            best_by_label[label] = dict(prediction)
            best_by_label[label]["score"] = round(score, 4)

    deduplicated = list(best_by_label.values())

    deduplicated.sort(
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )

    return deduplicated


def postprocess_temporal_actions(
    frame_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Recebe os resultados de Actions de vários frames de uma mesma cena/trecho
    e gera labels temporais mais estáveis.

    A saída prioriza labels úteis para audiodescrição.
    Exemplo: em uma cena de corrida, "sports" e "exercising" podem ser
    semanticamente possíveis, mas são secundárias. A descrição geralmente deve
    preferir "running", "moving" e "fast_motion".
    """

    label_scores: Dict[str, float] = {}
    label_counts: Counter = Counter()

    total_frames = len(frame_results)

    for frame_result in frame_results:
        predictions = frame_result.get("predictions", [])

        for prediction in predictions:
            label = prediction.get("label")
            score = float(prediction.get("score", 0.0))

            if not label:
                continue

            label_counts[label] += 1
            label_scores[label] = max(label_scores.get(label, 0.0), score)

    final_predictions: List[Dict[str, Any]] = []

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
                "frame_count": min(
                    total_frames,
                    label_counts["fast_motion"] + label_counts["moving"],
                ),
            }
        )

    elif has_fast_motion and total_frames >= 3:
        final_predictions.append(
            {
                "label": "running",
                "score": 0.65,
                "source": "temporal_rule",
                "frame_count": min(total_frames, label_counts["fast_motion"]),
            }
        )

    elif has_moving and has_exercising:
        final_predictions.append(
            {
                "label": "running",
                "score": 0.6,
                "source": "temporal_rule",
                "frame_count": min(
                    total_frames,
                    label_counts["moving"] + label_counts["exercising"],
                ),
            }
        )

    final_predictions = deduplicate_predictions(final_predictions)

    has_running = any(
        item["label"] == "running"
        for item in final_predictions
    )

    has_specific_sport_context = any(
        label_counts[label] > 0
        for label in [
            "ball_sport",
            "racket_sport",
            "water_activity",
            "martial_activity",
            "swimming",
            "cycling",
        ]
    )

    if has_running:
        cleaned_predictions = []

        for item in final_predictions:
            label = item["label"]
            score = float(item.get("score", 0.0))
            frame_count = int(item.get("frame_count", 0))

            # Em cena de corrida, estas labels tendem a ser ruído visual de passada.
            if label in ["standing", "still", "jumping", "falling", "throwing", "arms_raised"]:
                continue

            # sports só é mantido se houver evidência esportiva específica.
            if label == "sports" and not has_specific_sport_context:
                continue

            # exercising é plausível em corrida, mas só vale entrar na narrativa
            # se for recorrente e forte. Caso contrário, running já comunica melhor.
            if label == "exercising":
                is_recurrent = frame_count >= max(2, total_frames // 2)
                is_strong = score >= 0.80

                if not (is_recurrent and is_strong):
                    continue

            cleaned_predictions.append(item)

        final_predictions = cleaned_predictions

    else:
        # Se não há corrida temporal, ainda removemos ruídos muito fracos.
        final_predictions = [
            item
            for item in final_predictions
            if not (item["label"] == "falling" and float(item.get("score", 0.0)) < 0.70)
        ]

    final_predictions = deduplicate_predictions(final_predictions)

    return {
        "task_name": "action_temporal",
        "source": "temporal_postprocess",
        "frames_analyzed": total_frames,
        "predictions": final_predictions,
    }
