import argparse
from pathlib import Path
from typing import Dict, List, Optional

import fiftyone.zoo as foz
import pandas as pd
from PIL import Image

from ai.spectra.labels.label_sets import SPECTRA_OBJECT_LABELS


COCO_TO_SPECTRA_OBJECT: Dict[str, str] = {
    "person": "person" if "person" in SPECTRA_OBJECT_LABELS else None,

    "book": "book",
    "chair": "chair",
    "couch": "sofa",
    "bed": "bed",
    "dining table": "table",

    "cell phone": "phone",
    "laptop": "computer",
    "tv": "television",

    "car": "car",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "bus": "bus",

    "dog": "dog",
    "cat": "cat",

    "cup": "cup",
    "bottle": "bottle",
    "knife": "knife",

    "backpack": "backpack",
    "handbag": "bag",
    "suitcase": "bag",

    "sports ball": "ball",

    "keyboard": "computer",
    "mouse": "computer",
    "remote": "phone",

    "fork": "knife",
    "spoon": "cup",

    "bowl": "food",
    "banana": "food",
    "apple": "food",
    "sandwich": "food",
    "orange": "food",
    "broccoli": "food",
    "carrot": "food",
    "hot dog": "food",
    "pizza": "food",
    "donut": "food",
    "cake": "food",

    "teddy bear": "toy",
}


OBJECT_CLASS_PRIORITY: List[str] = [
    "book",
    "chair",
    "couch",
    "bed",
    "dining table",
    "cell phone",
    "laptop",
    "tv",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "dog",
    "cat",
    "cup",
    "bottle",
    "knife",
    "backpack",
    "handbag",
    "suitcase",
    "sports ball",
    "keyboard",
    "mouse",
    "remote",
    "fork",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "teddy bear",
]


def normalize_mapping(mapping: Dict[str, Optional[str]]) -> Dict[str, str]:
    normalized = {}

    for external_label, spectra_label in mapping.items():
        if spectra_label is None:
            continue

        if spectra_label not in SPECTRA_OBJECT_LABELS:
            continue

        normalized[external_label] = spectra_label

    return normalized


def get_supported_coco_classes(mapping: Dict[str, str]) -> List[str]:
    return [
        coco_label
        for coco_label in OBJECT_CLASS_PRIORITY
        if coco_label in mapping
    ]


def crop_detection(
    image_path: Path,
    bounding_box: List[float],
    output_path: Path,
    padding_ratio: float = 0.04,
    min_crop_size: int = 32,
) -> bool:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    x, y, width, height = bounding_box

    x_min = x * image_width
    y_min = y * image_height
    x_max = (x + width) * image_width
    y_max = (y + height) * image_height

    box_width = x_max - x_min
    box_height = y_max - y_min

    if box_width < min_crop_size or box_height < min_crop_size:
        return False

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


def build_object_row(
    crop_path: Path,
    active_label: str,
    source_dataset: str,
    source_sample_path: str,
    source_label: str,
    source_sample_id: str,
) -> Dict[str, object]:
    row = {
        "frame_path": str(crop_path),
        "source_dataset": source_dataset,
        "source_sample_path": source_sample_path,
        "source_label": source_label,
        "source_sample_id": source_sample_id,
    }

    for label in SPECTRA_OBJECT_LABELS:
        row[label] = 1 if label == active_label else 0

    return row


def should_skip_label(
    spectra_label: str,
    label_counts: Dict[str, int],
    max_per_label: int,
) -> bool:
    if spectra_label not in label_counts:
        return True

    return label_counts[spectra_label] >= max_per_label


def load_coco_dataset(
    split: str,
    classes: List[str],
    max_samples: int,
    dataset_name: str,
):
    return foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
        dataset_name=dataset_name,
    )


def build_spectra_object_dataset_from_fiftyone(
    output_dir: str,
    output_csv: str,
    split: str = "train",
    max_samples: int = 5000,
    max_per_label: int = 500,
    min_crop_size: int = 40,
    padding_ratio: float = 0.04,
    dataset_name: str = "spectra_object_coco_subset",
) -> Path:
    output_dir = Path(output_dir)
    output_csv = Path(output_csv)

    coco_to_spectra = normalize_mapping(COCO_TO_SPECTRA_OBJECT)
    coco_classes = get_supported_coco_classes(coco_to_spectra)

    if not coco_classes:
        raise ValueError(
            "Nenhuma classe COCO é compatível com SPECTRA_OBJECT_LABELS."
        )

    dataset = load_coco_dataset(
        split=split,
        classes=coco_classes,
        max_samples=max_samples,
        dataset_name=dataset_name,
    )

    label_counts = {
        label: 0
        for label in SPECTRA_OBJECT_LABELS
    }

    rows = []

    for sample in dataset:
        if not hasattr(sample, "ground_truth") or sample.ground_truth is None:
            continue

        detections = sample.ground_truth.detections

        for detection_index, detection in enumerate(detections):
            coco_label = detection.label

            if coco_label not in coco_to_spectra:
                continue

            spectra_label = coco_to_spectra[coco_label]

            if should_skip_label(
                spectra_label=spectra_label,
                label_counts=label_counts,
                max_per_label=max_per_label,
            ):
                continue

            source_image_path = Path(sample.filepath)

            if not source_image_path.exists():
                continue

            crop_name = "{}_{}_{}.jpg".format(
                source_image_path.stem,
                spectra_label,
                detection_index,
            )

            crop_path = (
                output_dir
                / "crops"
                / spectra_label
                / crop_name
            )

            success = crop_detection(
                image_path=source_image_path,
                bounding_box=detection.bounding_box,
                output_path=crop_path,
                padding_ratio=padding_ratio,
                min_crop_size=min_crop_size,
            )

            if not success:
                continue

            rows.append(
                build_object_row(
                    crop_path=crop_path,
                    active_label=spectra_label,
                    source_dataset="coco-2017",
                    source_sample_path=str(source_image_path),
                    source_label=coco_label,
                    source_sample_id=str(sample.id),
                )
            )

            label_counts[spectra_label] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_csv, index=False)

    print("\nDataset de objetos criado.")
    print("CSV:", output_csv)
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
        description="Cria dataset automático para SpectraObjectNet usando FiftyOne + COCO 2017."
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
        "--max-per-label",
        type=int,
        default=500,
        help="Quantidade máxima de crops por label da Spectra.",
    )

    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=40,
        help="Tamanho mínimo em pixels para largura e altura do crop.",
    )

    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.04,
        help="Margem extra aplicada ao redor da bounding box.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/dataset_sources/spectra_object_fiftyone",
        help="Diretório onde os crops serão salvos.",
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/spectra_object_coco_fiftyone_labels.csv",
        help="CSV de saída no formato esperado pela Spectra.",
    )

    parser.add_argument(
        "--dataset-name",
        default="spectra_object_coco_subset",
        help="Nome interno do dataset no FiftyOne.",
    )

    args = parser.parse_args()

    build_spectra_object_dataset_from_fiftyone(
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        split=args.split,
        max_samples=args.max_samples,
        max_per_label=args.max_per_label,
        min_crop_size=args.min_crop_size,
        padding_ratio=args.padding_ratio,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()



# python3 -m tools.build_spectra_object_dataset_fiftyone \
#   --split train \
#   --max-samples 8000 \
#   --max-per-label 600 \
#   --min-crop-size 40 \
#   --padding-ratio 0.04 \
#   --output-dir data/dataset_sources/spectra_object_fiftyone \
#   --output-csv data/datasets/spectra_object_labels.csv \
#   --dataset-name spectra_object_coco_train