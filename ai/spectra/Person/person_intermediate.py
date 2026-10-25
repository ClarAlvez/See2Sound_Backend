"""Verificador experimental de atributos Person por recorte, sem alterar pesos.

O CLIP fornece scores relativos a pares de prompts; não probabilidades
calibradas. Só os fatos narrativos filtrados são modificados.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any
import os


MODEL_ID = "openai/clip-vit-base-patch32"
MIN_YOLO_SCORE = 0.45
MIN_SPECTRA_SCORE = 0.50
MAX_REMOVE_SPECTRA_SCORE = 0.75
MIN_NEGATIVE_MARGIN = 0.22
MIN_POSITIVE_MARGIN = 0.10
MAX_DESCRIBED_PEOPLE = 3

# Apenas propriedades de aparência; nunca inferir identidade, idade ou gênero.
PROMPTS = {
    "glasses": ("a person wearing eyeglasses", "a person without eyeglasses"),
    "hat": ("a person wearing a hat", "a person with no hat"),
    "backpack": ("a person wearing a backpack", "a person with no backpack"),
    "handbag": ("a person holding a handbag", "a person without a handbag"),
    "long_hair": ("a person with long hair", "a person with short hair"),
    "short_hair": ("a person with short hair", "a person with long hair"),
    "black_clothes": ("a person wearing black clothing", "a person wearing colorful or light clothing"),
    "white_clothes": ("a person wearing white clothing", "a person wearing dark or colorful clothing"),
    "red_clothes": ("a person wearing red clothing", "a person wearing clothing that is not red"),
    "blue_clothes": ("a person wearing blue clothing", "a person wearing clothing that is not blue"),
    "green_clothes": ("a person wearing green clothing", "a person wearing clothing that is not green"),
}

EXCLUSIVE = ({"short_hair", "long_hair"},)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class PersonCLIPVerifier:
    def __init__(self, model_id: str = MODEL_ID):
        import torch
        from transformers import AutoProcessor, CLIPModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.model_id = model_id

    def verify(self, crop_path: str, labels: list[str]) -> dict[str, dict]:
        from PIL import Image
        selected = [key for key in labels if key in PROMPTS]
        if not selected:
            return {}
        path = Path(crop_path)
        if not path.is_file():
            return {}
        with Image.open(path) as img:
            image = img.convert("RGB")
            texts = [p for label in selected for p in PROMPTS[label]]
            batch = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.inference_mode():
                logits = self.model(**batch).logits_per_image[0]
            # Softmax realizado SOMENTE em cada par positivo/negativo.
            pair_scores = logits.reshape(-1, 2).softmax(dim=1).cpu().tolist()
        return {
            label: {
                "positive_relative_score": round(float(scores[0]), 4),
                "negative_relative_score": round(float(scores[1]), 4),
                "margin": round(float(scores[0] - scores[1]), 4),
                "calibrated": False,
            }
            for label, scores in zip(selected, pair_scores)
        }


def _predictions(result: Any) -> dict[str, float]:
    if not isinstance(result, dict):
        return {}
    return {
        p["label"]: _num(p.get("score"))
        for p in result.get("predictions", [])
        if isinstance(p, dict) and isinstance(p.get("label"), str)
    }


def _frame_people(frame: dict, verifier: PersonCLIPVerifier | None) -> list[dict]:
    raw = frame.get("raw") or {}
    person_raw = raw.get("person") or {}
    crops = person_raw.get("raw_crop_results") or []
    if not crops:
        crops = [{"crop": crop, "result": {}} for crop in raw.get("person_crops", [])]
    people = []
    for item in crops:
        crop = item.get("crop") or {}
        score = _num(crop.get("detector_confidence"))
        bbox = crop.get("detection_bbox") or crop.get("bbox")
        if score < MIN_YOLO_SCORE or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        original = _predictions(item.get("result"))
        original_attrs = {k: v for k, v in original.items() if k in PROMPTS and v >= MIN_SPECTRA_SCORE}
        checks = verifier.verify(str(crop.get("crop_path", "")), list(original_attrs)) if verifier else {}
        decisions = {}
        kept = {}
        for label, original_score in original_attrs.items():
            info = checks.get(label)
            margin = info["margin"] if info else None
            # Só rejeitar com evidência negativa forte E score Spectra não alto.
            if margin is not None and margin <= -MIN_NEGATIVE_MARGIN and original_score < MAX_REMOVE_SPECTRA_SCORE:
                decision = "remove_from_narrative"
            elif margin is not None and margin >= MIN_POSITIVE_MARGIN:
                decision = "confirmed"
            else:
                decision = "uncertain_keep"
            decisions[label] = {"decision": decision, "spectra_score": round(original_score, 4), "clip": info}
            if decision != "remove_from_narrative":
                kept[label] = original_score
        for group in EXCLUSIVE:
            matches = [k for k in group if k in kept]
            if len(matches) > 1:
                best = max(matches, key=kept.get)
                for label in matches:
                    if label != best:
                        del kept[label]
                        decisions[label]["decision"] = "remove_conflicting_attribute"
        x1, y1, x2, y2 = map(float, bbox)
        people.append({
            "bbox": list(map(float, bbox)),
            "bbox_area": max(0.0, x2-x1) * max(0.0, y2-y1),
            "detector_score": round(score, 4),
            "attributes": [{"label": k, "score": round(v, 4)} for k, v in sorted(kept.items(), key=lambda kv: -kv[1])[:3]],
            "decisions": decisions,
        })
    return sorted(people, key=lambda p: -p["bbox_area"])


def analyze_person_scene(scene: dict, verifier: PersonCLIPVerifier | None = None,
                         detection_cap: int = 3) -> dict:
    samples = []
    for frame in scene.get("raw_frame_results", []):
        samples.append({"timestamp": frame.get("timestamp"), "people": _frame_people(frame, verifier)})
    counts = [len(sample["people"]) for sample in samples]
    positives = [count for count in counts if count]
    if not positives:
        return {"status": "no_confirmed_people", "count_type": "unknown", "count": 0,
                "frame_counts": counts, "people": [], "source": "yolo_person_crops_clip"}
    typical = int(median(positives) + 0.5)
    popular, support = Counter(counts).most_common(1)[0]
    stable = len(counts) >= 2 and popular > 0 and support / len(counts) >= 0.60
    # Contagem == cap pode indicar truncamento, nunca declarar exatidão.
    count_type = "group" if typical >= detection_cap else "exact" if stable else "uncertain"
    representative = min(
        samples,
        key=lambda s: (abs(len(s["people"]) - typical), -sum(p["detector_score"] for p in s["people"])),
    )
    return {
        "status": "analyzed", "count": typical, "count_type": count_type,
        "frame_counts": counts, "count_consistent": stable,
        "representative_timestamp": representative["timestamp"],
        "people": representative["people"][:MAX_DESCRIBED_PEOPLE],
        "source": "yolo_person_crops_clip", "clip_model": verifier.model_id if verifier else None,
        "tracking_enabled": False, "count_capped": typical >= detection_cap,
    }


def apply_person_intermediate(scene: dict, verifier: PersonCLIPVerifier | None = None,
                              detection_cap: int = 3) -> dict:
    result = deepcopy(scene)
    analysis = analyze_person_scene(scene, verifier=verifier, detection_cap=detection_cap)
    result["person_intermediate"] = analysis
    result.setdefault("context", {})["person_intermediate"] = analysis
    # Não sobrescrever labels globais: elas misturam diferentes pessoas.
    # O adaptador narrativo deve usar SOMENTE attributes de cada recorte.
    return result
