import argparse
from pathlib import Path
from typing import Dict, List, Optional

import fiftyone.zoo as foz
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ai.spectra.labels.label_sets import SPECTRA_PERSON_LABELS


PERSON_ATTRIBUTE_PROMPTS: Dict[str, str] = {
    "person": "a cropped image of a person",
    "face_visible": "a cropped image of a person with a visible face",
    "hand_visible": "a cropped image of a person with visible hands",

    "man": "a cropped image of a man",
    "woman": "a cropped image of a woman",
    "child": "a cropped image of a child",

    "blonde_hair": "a person with blonde hair",
    "brown_hair": "a person with brown hair",
    "black_hair": "a person with black hair",
    "red_hair": "a person with red hair",
    "gray_hair": "a person with gray hair",

    "short_hair": "a person with short hair",
    "long_hair": "a person with long hair",
    "curly_hair": "a person with curly hair",
    "straight_hair": "a person with straight hair",

    "red_clothes": "a person wearing red clothes",
    "blue_clothes": "a person wearing blue clothes",
    "black_clothes": "a person wearing black clothes",
    "white_clothes": "a person wearing white clothes",
    "green_clothes": "a person wearing green clothes",
    "yellow_clothes": "a person wearing yellow clothes",

    "dress": "a person wearing a dress",
    "shirt": "a person wearing a shirt",
    "jacket": "a person wearing a jacket",

    "glasses": "a person wearing glasses",
    "hat": "a person wearing a hat",
    "cap": "a person wearing a cap",
    "backpack": "a person wearing a backpack",
    "bag": "a person carrying a bag",

    "light_skin": "a person with light skin tone",
    "medium_skin": "a person with medium skin tone",
    "dark_skin": "a person with dark skin tone",
}


PERSON_ATTRIBUTE_THRESHOLDS: Dict[str, float] = {
    "person": 0.50,
    "face_visible": 0.62,
    "hand_visible": 0.65,

    "man": 0.68,
    "woman": 0.68,
    "child": 0.72,

    "blonde_hair": 0.72,
    "brown_hair": 0.72,
    "black_hair": 0.72,
    "red_hair": 0.76,
    "gray_hair": 0.76,

    "short_hair": 0.70,
    "long_hair": 0.70,
    "curly_hair": 0.74,
    "straight_hair": 0.74,

    "red_clothes": 0.68,
    "blue_clothes": 0.68,
    "black_clothes": 0.68,
    "white_clothes": 0.68,
    "green_clothes": 0.70,
    "yellow_clothes": 0.70,

    "dress": 0.70,
    "shirt": 0.66,
    "jacket": 0.68,

    "glasses": 0.72,
    "hat": 0.70,
    "cap": 0.70,
    "backpack": 0.72,
    "bag": 0.72,

    "light_skin": 0.80,
    "medium_skin": 0.80,
    "dark_skin": 0.80,
}


MUTUALLY_EXCLUSIVE_PERSON_GROUPS: List[List[str]] = [
    ["man", "woman", "child"],
    ["blonde_hair", "brown_hair", "black_hair", "red_hair", "gray_hair"],
    ["short_hair", "long_hair"],
    ["curly_hair", "straight_hair"],
    ["light_skin", "medium_skin", "dark_skin"],
]


def crop_detection(
    image_path: Path,
    bounding_box: List[float],
    output_path: Path,
    padding_ratio: float = 0.08,
    min_crop_height_ratio: float = 0.20,
) -> bool:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    x, y, width, height = bounding_box

    if height < min_crop_height_ratio:
        return False

    x_min = x * image_width
    y_min = y * image_height
    x_max = (x + width) * image_width
    y_max = (y + height) * image_height

    box_width = x_max - x_min
    box_height = y_max - y_min

    x_min -= box_width * padding_ratio
    x_max += box_width * padding_ratio
    y_min -= box_height * padding_ratio
    y_max += box_height * padding_ratio

    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(image_width, int(x_max))
    y_max = min(image_height, int(y_max))

    if x_max <= x_min or y_max <= y_min:
        return False

    crop = image.crop((x_min, y_min, x_max, y_max))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)

    return True


class ClipPersonAttributeLabeler:
    def __init__(
        self,
        device: Optional[str] = None,
        margin_scale: float = 50.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.margin_scale = margin_scale

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

        self.model.eval()

        self.text_features_by_label = self._build_text_features()

    def _build_text_features(self) -> Dict[str, torch.Tensor]:
        text_features_by_label = {}

        with torch.no_grad():
            for label in SPECTRA_PERSON_LABELS:
                positive_prompt = PERSON_ATTRIBUTE_PROMPTS.get(
                    label,
                    label.replace("_", " "),
                )

                negative_prompt = "a cropped image without {}".format(
                    positive_prompt
                )

                inputs = self.processor(
                    text=[positive_prompt, negative_prompt],
                    return_tensors="pt",
                    padding=True,
                )

                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }

                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(
                    dim=-1,
                    keepdim=True,
                )

                text_features_by_label[label] = text_features

        return text_features_by_label

    @torch.no_grad()
    def label_crop(self, crop_path: Path):
        image = Image.open(crop_path).convert("RGB")

        image_inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        image_inputs = {
            key: value.to(self.device)
            for key, value in image_inputs.items()
        }

        image_features = self.model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True,
        )

        labels = {
            label: 0
            for label in SPECTRA_PERSON_LABELS
        }

        scores = {
            label: 0.0
            for label in SPECTRA_PERSON_LABELS
        }

        if "person" in labels:
            labels["person"] = 1
            scores["person"] = 1.0

        for label in SPECTRA_PERSON_LABELS:
            if label == "person":
                continue

            text_features = self.text_features_by_label[label]
            similarities = (image_features @ text_features.T).squeeze(0).cpu()

            positive_similarity = float(similarities[0])
            negative_similarity = float(similarities[1])

            margin = positive_similarity - negative_similarity

            confidence = torch.sigmoid(
                torch.tensor(margin * self.margin_scale)
            ).item()

            scores[label] = round(confidence, 4)

            threshold = PERSON_ATTRIBUTE_THRESHOLDS.get(label, 0.70)

            if confidence > threshold:
                labels[label] = 1

        self._apply_consistency_rules(labels, scores)

        return labels, scores

    def _apply_consistency_rules(
        self,
        labels: Dict[str, int],
        scores: Dict[str, float],
    ) -> None:
        def keep_only_best(group: List[str]) -> None:
            active_labels = [
                label
                for label in group
                if labels.get(label, 0) == 1
            ]

            if len(active_labels) <= 1:
                return

            best_label = max(
                active_labels,
                key=lambda label: scores.get(label, 0.0),
            )

            for label in active_labels:
                labels[label] = 1 if label == best_label else 0

        for group in MUTUALLY_EXCLUSIVE_PERSON_GROUPS:
            keep_only_best(group)

        if labels.get("person", 0) == 0:
            for label in labels:
                labels[label] = 0


def build_person_row(
    crop_path: Path,
    labels: Dict[str, int],
    source_dataset: str,
    source_sample_path: str,
    source_sample_id: str,
) -> Dict[str, object]:
    row = {
        "frame_path": str(crop_path),
        "source_dataset": source_dataset,
        "source_sample_path": source_sample_path,
        "source_sample_id": source_sample_id,
    }

    for label in SPECTRA_PERSON_LABELS:
        row[label] = int(labels.get(label, 0))

    return row


def build_person_score_row(
    crop_path: Path,
    scores: Dict[str, float],
    source_dataset: str,
    source_sample_path: str,
    source_sample_id: str,
) -> Dict[str, object]:
    row = {
        "frame_path": str(crop_path),
        "source_dataset": source_dataset,
        "source_sample_path": source_sample_path,
        "source_sample_id": source_sample_id,
    }

    for label in SPECTRA_PERSON_LABELS:
        row[label] = float(scores.get(label, 0.0))

    return row


def load_coco_person_dataset(
    split: str,
    max_samples: int,
    dataset_name: str,
):
    return foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=["person"],
        max_samples=max_samples,
        dataset_name=dataset_name,
    )


def build_spectra_person_dataset_from_fiftyone(
    output_dir: str,
    output_csv: str,
    output_scores_csv: str,
    split: str = "train",
    max_samples: int = 5000,
    max_crops: int = 3000,
    min_crop_height_ratio: float = 0.20,
    padding_ratio: float = 0.08,
    dataset_name: str = "spectra_person_coco_subset",
    pseudo_label_attributes: bool = True,
    margin_scale: float = 50.0,
) -> Path:
    output_dir = Path(output_dir)
    output_csv = Path(output_csv)
    output_scores_csv = Path(output_scores_csv)

    dataset = load_coco_person_dataset(
        split=split,
        max_samples=max_samples,
        dataset_name=dataset_name,
    )

    labeler = None

    if pseudo_label_attributes:
        labeler = ClipPersonAttributeLabeler(
            margin_scale=margin_scale,
        )

    rows = []
    score_rows = []
    crop_count = 0

    label_counts = {
        label: 0
        for label in SPECTRA_PERSON_LABELS
    }

    for sample in dataset:
        if crop_count >= max_crops:
            break

        if not hasattr(sample, "ground_truth") or sample.ground_truth is None:
            continue

        detections = sample.ground_truth.detections

        for detection_index, detection in enumerate(detections):
            if crop_count >= max_crops:
                break

            if detection.label != "person":
                continue

            source_image_path = Path(sample.filepath)

            if not source_image_path.exists():
                continue

            crop_name = "{}_person_{}.jpg".format(
                source_image_path.stem,
                crop_count,
            )

            crop_path = output_dir / "crops" / "person" / crop_name

            success = crop_detection(
                image_path=source_image_path,
                bounding_box=detection.bounding_box,
                output_path=crop_path,
                padding_ratio=padding_ratio,
                min_crop_height_ratio=min_crop_height_ratio,
            )

            if not success:
                continue

            if labeler is not None:
                labels, scores = labeler.label_crop(crop_path)
            else:
                labels = {
                    label: 0
                    for label in SPECTRA_PERSON_LABELS
                }

                scores = {
                    label: 0.0
                    for label in SPECTRA_PERSON_LABELS
                }

                if "person" in labels:
                    labels["person"] = 1
                    scores["person"] = 1.0

            rows.append(
                build_person_row(
                    crop_path=crop_path,
                    labels=labels,
                    source_dataset="coco-2017",
                    source_sample_path=str(source_image_path),
                    source_sample_id=str(sample.id),
                )
            )

            score_rows.append(
                build_person_score_row(
                    crop_path=crop_path,
                    scores=scores,
                    source_dataset="coco-2017",
                    source_sample_path=str(source_image_path),
                    source_sample_id=str(sample.id),
                )
            )

            for label, value in labels.items():
                if int(value) == 1 and label in label_counts:
                    label_counts[label] += 1

            crop_count += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_scores_csv.parent.mkdir(parents=True, exist_ok=True)

    labels_dataframe = pd.DataFrame(rows)
    scores_dataframe = pd.DataFrame(score_rows)

    labels_dataframe.to_csv(output_csv, index=False)
    scores_dataframe.to_csv(output_scores_csv, index=False)

    print("\nDataset de pessoas criado.")
    print("CSV:", output_csv)
    print("CSV de scores:", output_scores_csv)
    print("Total de crops:", len(rows))
    print("\nDistribuição por label:")

    for label, count in sorted(
        label_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        if count > 0:
            print("{}: {}".format(label, count))

    zero_labels = [
        label
        for label, count in label_counts.items()
        if count == 0
    ]

    print("\nLabels sem exemplos:")
    print(zero_labels)

    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Cria dataset automático para SpectraPersonNet usando FiftyOne + COCO 2017."
    )

    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Split do COCO 2017 usado pelo FiftyOne.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Quantidade máxima de imagens carregadas do COCO.",
    )

    parser.add_argument(
        "--max-crops",
        type=int,
        default=3000,
        help="Quantidade máxima de crops de pessoas.",
    )

    parser.add_argument(
        "--min-crop-height-ratio",
        type=float,
        default=0.20,
        help="Altura mínima da bounding box em relação à altura total da imagem.",
    )

    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.08,
        help="Margem extra aplicada ao redor da bounding box da pessoa.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/dataset_sources/spectra_person_fiftyone",
        help="Diretório onde os crops serão salvos.",
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/spectra_person_coco_fiftyone_labels.csv",
        help="CSV de saída no formato esperado pela Spectra.",
    )

    parser.add_argument(
        "--output-scores-csv",
        default="data/datasets/spectra_person_coco_fiftyone_scores.csv",
        help="CSV de scores das pseudo-labels.",
    )

    parser.add_argument(
        "--dataset-name",
        default="spectra_person_coco_subset",
        help="Nome interno do dataset no FiftyOne.",
    )

    parser.add_argument(
        "--disable-pseudo-label",
        action="store_true",
        help="Desativa pseudo-label de atributos e marca apenas person=1.",
    )

    parser.add_argument(
        "--margin-scale",
        type=float,
        default=50.0,
        help="Escala aplicada na margem entre prompt positivo e negativo do CLIP.",
    )

    args = parser.parse_args()

    build_spectra_person_dataset_from_fiftyone(
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        output_scores_csv=args.output_scores_csv,
        split=args.split,
        max_samples=args.max_samples,
        max_crops=args.max_crops,
        min_crop_height_ratio=args.min_crop_height_ratio,
        padding_ratio=args.padding_ratio,
        dataset_name=args.dataset_name,
        pseudo_label_attributes=not args.disable_pseudo_label,
        margin_scale=args.margin_scale,
    )


if __name__ == "__main__":
    main()

# python3 -m tools.build_spectra_person_dataset_fiftyone \
#   --split train \
#   --max-samples 8000 \
#   --max-crops 5000 \
#   --min-crop-height-ratio 0.20 \
#   --padding-ratio 0.08 \
#   --output-dir data/dataset_sources/spectra_person_fiftyone \
#   --output-csv data/datasets/spectra_person_labels.csv \
#   --output-scores-csv data/datasets/spectra_person_scores.csv \
#   --dataset-name spectra_person_coco_train \
#   --margin-scale 50