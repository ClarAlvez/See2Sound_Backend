import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fiftyone.zoo as foz
import pandas as pd
from PIL import Image

from ai.spectra.Object.labels import SPECTRA_OBJECT_LABELS


"""
Builder automático para SpectraObjectNet usando FiftyOne + COCO 2017.

Objetivo:
- Criar crops limpos de objetos.
- Evitar mapeamentos semânticos perigosos.
- Gerar CSV no formato esperado pelo train.py.
- Manter apenas labels confiáveis para treino inicial da ObjectNet.

Observação:
Este builder é propositalmente mais rígido do que a versão anterior.
É melhor gerar menos dados mais confiáveis do que muitos crops ruins.
"""


# ============================================================
# Mapeamento COCO -> SpectraObjectNet
# ============================================================

# Mantém apenas mapeamentos diretos ou muito seguros.
# Evita coisas como:
# - fork -> knife
# - spoon -> cup
# - remote -> phone
# - keyboard/mouse -> computer
# porque isso suja o treino.
COCO_TO_SPECTRA_OBJECT: Dict[str, Optional[str]] = {
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

    "teddy bear": "toy",

    # Comida: mapeamento aceito, mas pode ser desativado por flag.
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
    "teddy bear",

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
]


# Algumas labels são mais perigosas porque costumam gerar crops ruins
# ou podem ter ambiguidade visual. Exigimos área mínima maior para elas.
LABEL_MIN_AREA_RATIO: Dict[str, float] = {
    "phone": 0.010,
    "knife": 0.010,
    "book": 0.012,
    "cup": 0.008,
    "bottle": 0.008,
    "bag": 0.014,
    "backpack": 0.014,
    "food": 0.012,
    "toy": 0.012,
}


# Limite mínimo padrão de área do crop em relação à imagem inteira.
DEFAULT_MIN_AREA_RATIO = 0.006


# Evita crops exagerados que provavelmente representam contexto inteiro,
# não o objeto isolado.
DEFAULT_MAX_AREA_RATIO = 0.75


def normalize_mapping(
    mapping: Dict[str, Optional[str]],
    include_food: bool,
) -> Dict[str, str]:
    normalized = {}

    for external_label, spectra_label in mapping.items():
        if spectra_label is None:
            continue

        if spectra_label == "food" and not include_food:
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


def get_bbox_pixels(
    image_width: int,
    image_height: int,
    bounding_box: List[float],
    padding_ratio: float,
) -> Optional[Tuple[int, int, int, int]]:
    x, y, width, height = bounding_box

    x_min = x * image_width
    y_min = y * image_height
    x_max = (x + width) * image_width
    y_max = (y + height) * image_height

    box_width = x_max - x_min
    box_height = y_max - y_min

    if box_width <= 0 or box_height <= 0:
        return None

    x_min -= box_width * padding_ratio
    x_max += box_width * padding_ratio
    y_min -= box_height * padding_ratio
    y_max += box_height * padding_ratio

    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(image_width, int(x_max))
    y_max = min(image_height, int(y_max))

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def is_valid_crop(
    spectra_label: str,
    image_width: int,
    image_height: int,
    bbox_pixels: Tuple[int, int, int, int],
    min_crop_size: int,
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> bool:
    x_min, y_min, x_max, y_max = bbox_pixels

    crop_width = x_max - x_min
    crop_height = y_max - y_min

    if crop_width < min_crop_size or crop_height < min_crop_size:
        return False

    image_area = image_width * image_height
    crop_area = crop_width * crop_height

    if image_area <= 0:
        return False

    area_ratio = crop_area / image_area

    label_min_area = LABEL_MIN_AREA_RATIO.get(
        spectra_label,
        min_area_ratio,
    )

    if area_ratio < label_min_area:
        return False

    if area_ratio > max_area_ratio:
        return False

    aspect_ratio = crop_width / max(crop_height, 1)

    if aspect_ratio < min_aspect_ratio:
        return False

    if aspect_ratio > max_aspect_ratio:
        return False

    return True


def crop_detection(
    image_path: Path,
    bounding_box: List[float],
    output_path: Path,
    spectra_label: str,
    padding_ratio: float,
    min_crop_size: int,
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> bool:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    bbox_pixels = get_bbox_pixels(
        image_width=image_width,
        image_height=image_height,
        bounding_box=bounding_box,
        padding_ratio=padding_ratio,
    )

    if bbox_pixels is None:
        return False

    if not is_valid_crop(
        spectra_label=spectra_label,
        image_width=image_width,
        image_height=image_height,
        bbox_pixels=bbox_pixels,
        min_crop_size=min_crop_size,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    ):
        return False

    crop = image.crop(bbox_pixels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=95)

    return True


def build_object_row(
    crop_path: Path,
    active_label: str,
    source_dataset: str,
    source_sample_path: str,
    source_label: str,
    source_sample_id: str,
    detection_index: int,
) -> Dict[str, object]:
    row = {
        "frame_path": str(crop_path),
        "source_dataset": source_dataset,
        "source_sample_path": source_sample_path,
        "source_label": source_label,
        "source_sample_id": source_sample_id,
        "source_detection_index": detection_index,
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
    max_per_label: int = 300,
    min_crop_size: int = 64,
    padding_ratio: float = 0.02,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
    min_aspect_ratio: float = 0.20,
    max_aspect_ratio: float = 5.00,
    include_food: bool = False,
    dataset_name: str = "spectra_object_coco_strict_subset",
) -> Path:
    output_dir = Path(output_dir)
    output_csv = Path(output_csv)

    coco_to_spectra = normalize_mapping(
        mapping=COCO_TO_SPECTRA_OBJECT,
        include_food=include_food,
    )

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
                spectra_label=spectra_label,
                padding_ratio=padding_ratio,
                min_crop_size=min_crop_size,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
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
                    detection_index=detection_index,
                )
            )

            label_counts[spectra_label] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_csv, index=False)

    print("\nDataset de objetos criado.")
    print("CSV:", output_csv)
    print("Total de crops:", len(rows))
    print("Média esperada de labels positivas por crop: 1.0")
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
        description="Cria dataset rígido para SpectraObjectNet usando FiftyOne + COCO 2017."
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
        default=300,
        help="Quantidade máxima de crops por label da Spectra.",
    )

    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=64,
        help="Tamanho mínimo em pixels para largura e altura do crop.",
    )

    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.02,
        help="Margem extra aplicada ao redor da bounding box.",
    )

    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=DEFAULT_MIN_AREA_RATIO,
        help="Área mínima do crop em relação à imagem inteira.",
    )

    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=DEFAULT_MAX_AREA_RATIO,
        help="Área máxima do crop em relação à imagem inteira.",
    )

    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=0.20,
        help="Aspect ratio mínimo permitido para o crop.",
    )

    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=5.00,
        help="Aspect ratio máximo permitido para o crop.",
    )

    parser.add_argument(
        "--include-food",
        action="store_true",
        help="Inclui classes de comida do COCO mapeadas para food.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/dataset_sources/spectra_object_fiftyone",
        help="Diretório onde os crops serão salvos.",
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/spectra_object_labels.csv",
        help="CSV de saída no formato esperado pela Spectra.",
    )

    parser.add_argument(
        "--dataset-name",
        default="spectra_object_coco_strict_subset",
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
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        include_food=args.include_food,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
