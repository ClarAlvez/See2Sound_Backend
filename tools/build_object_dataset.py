import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
import json

from ai.spectra.Object.labels import LABELS


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


COCO_TO_SPECTRA_OBJECT = {
    # Furniture / indoor
    "chair": ["chair"],
    "couch": ["sofa"],
    "bed": ["bed"],
    "dining table": ["table"],
    "toilet": ["toilet"],
    "bench": ["chair"],

    # Electronics
    "cell phone": ["phone", "screen"],
    "laptop": ["computer", "screen"],
    "tv": ["television", "screen"],
    "keyboard": ["keyboard", "computer"],
    "mouse": ["mouse", "computer"],
    "remote": ["remote"],

    # Vehicles
    "car": ["car"],
    "bicycle": ["bicycle"],
    "motorcycle": ["motorcycle"],
    "bus": ["bus"],
    "truck": ["truck"],
    "train": ["train"],
    "boat": ["boat"],
    "airplane": ["airplane"],

    # Animals
    "dog": ["dog", "animal"],
    "cat": ["cat", "animal"],
    "bird": ["bird", "animal"],
    "horse": ["horse", "animal"],
    "sheep": ["sheep", "animal"],
    "cow": ["cow", "animal"],
    "elephant": ["animal"],
    "bear": ["animal"],
    "zebra": ["animal"],
    "giraffe": ["animal"],

    # Food / kitchen
    "banana": ["fruit", "food"],
    "apple": ["fruit", "food"],
    "orange": ["fruit", "food"],
    "broccoli": ["food"],
    "carrot": ["food"],
    "hot dog": ["food"],
    "pizza": ["food"],
    "donut": ["food"],
    "cake": ["food"],
    "sandwich": ["food"],
    "cup": ["cup"],
    "bottle": ["bottle"],
    "wine glass": ["cup"],
    "bowl": ["bowl"],
    "plate": ["plate"],
    "fork": ["fork"],
    "spoon": ["spoon"],
    "knife": ["knife"],

    # Bags / accessories
    "backpack": ["bag", "backpack"],
    "handbag": ["bag", "handbag"],
    "suitcase": ["bag", "suitcase"],
    "umbrella": ["umbrella"],

    # Reading / paper
    "book": ["book", "document"],

    # Sports / toys
    "sports ball": ["ball", "toy"],
    "frisbee": ["toy"],
    "kite": ["kite", "toy"],
    "skateboard": ["skateboard", "toy"],
    "surfboard": ["surfboard"],
    "tennis racket": ["sports_racket"],
    "baseball bat": ["sports_racket"],
    "baseball glove": ["sports_racket"],

    # Misc
    "teddy bear": ["toy"],
    "toothbrush": ["toy"],
    "scissors": ["knife"],

    # Ignorados por enquanto
    "person": [],
    "traffic light": [],
    "fire hydrant": [],
    "stop sign": [],
    "parking meter": [],
    "potted plant": [],
    "refrigerator": [],
    "oven": [],
    "microwave": [],
    "toaster": [],
    "sink": [],
    "vase": [],
    "clock": [],
    "hair drier": [],
    "tie": [],
    "skis": [],
    "snowboard": [],
}


def map_coco_label_to_object(label_name: str) -> List[str]:
    if label_name is None:
        return []

    label_name = normalize_text(label_name)
    mapped_labels = COCO_TO_SPECTRA_OBJECT.get(label_name, [])

    if mapped_labels is None:
        return []

    if isinstance(mapped_labels, str):
        mapped_labels = [mapped_labels]

    return sorted({
        label
        for label in mapped_labels
        if label in LABELS
    })


FOLDER_NAME_TO_SPECTRA_OBJECT: Dict[str, List[str]] = {
    **COCO_TO_SPECTRA_OBJECT,
    "cell_phone": ["phone", "screen"],
    "dining_table": ["table"],
    "sports_ball": ["ball"],
    "tennis_racket": ["sports_racket"],
    "baseball_bat": ["sports_racket"],
    "baseball_glove": ["ball"],
    "wine_glass": ["cup"],
    "teddy_bear": ["toy"],
    "couch": ["sofa"],
    "tv": ["screen", "television"],
    "laptop": ["computer", "screen"],
    "keyboard": ["keyboard", "computer"],
    "phone": ["phone", "screen"],
    "computer": ["computer", "screen"],
    "television": ["screen", "television"],
    "bag": ["bag"],
    "handbag": ["bag", "handbag"],
    "backpack": ["bag", "backpack"],
    "suitcase": ["bag", "suitcase"],
    "glasses": ["glasses"],
    "book": ["book", "document"],
    "paper": ["paper", "document"],
    "document": ["document", "paper"],
    "letter": ["letter", "paper", "document"],
    "photo": ["photo", "paper"],
    "key": ["key"],
    "weapon": ["weapon"],
    "blood": ["blood"],
    "dice": ["dice"],
    "miniature": ["miniature", "toy"],
    "board_game": ["board_game", "toy"],
    "musical_instrument": ["musical_instrument"],
    "instrument": ["musical_instrument"],
    "subtitles": ["subtitles", "on_screen_text"],
    "on_screen_text": ["on_screen_text"],
}


def normalize_text(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def normalize_folder_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_path_text(value: Any) -> str:
    return str(value).replace("\\", "/")

def command_from_manual_coco(args) -> None:
    coco_root = Path(args.coco_root)
    split = args.split

    if split == "train":
        image_dir = coco_root / "train2017"
        annotation_path = coco_root / "annotations" / "instances_train2017.json"
    elif split in ["validation", "val"]:
        image_dir = coco_root / "val2017"
        annotation_path = coco_root / "annotations" / "instances_val2017.json"
        split = "validation"
    else:
        raise ValueError("Split inválido. Use train ou validation.")

    if not image_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {image_dir}")

    if not annotation_path.exists():
        raise FileNotFoundError(f"Arquivo de anotações não encontrado: {annotation_path}")

    print("Lendo COCO manual...")
    print("Root:", coco_root)
    print("Split:", split)
    print("Imagens:", image_dir)
    print("Anotações:", annotation_path)

    with open(annotation_path, "r", encoding="utf-8") as file:
        coco_data = json.load(file)

    categories_by_id = {
        category["id"]: category["name"]
        for category in coco_data.get("categories", [])
    }

    images_by_id = {
        image["id"]: image["file_name"]
        for image in coco_data.get("images", [])
    }

    labels_by_image_id = {}

    mapped_category_counts = {}
    unmapped_category_counts = {}

    for annotation in coco_data.get("annotations", []):
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        category_name = categories_by_id.get(category_id)

        mapped_labels = map_coco_label_to_object(category_name)

        if not mapped_labels:
            if category_name is not None:
                unmapped_category_counts[category_name] = (
                    unmapped_category_counts.get(category_name, 0) + 1
                )
            continue

        if image_id not in labels_by_image_id:
            labels_by_image_id[image_id] = set()

        for mapped_label in mapped_labels:
            mapped_category_counts[mapped_label] = (
                mapped_category_counts.get(mapped_label, 0) + 1
            )
            labels_by_image_id[image_id].add(mapped_label)

    image_ids = list(images_by_id.keys())

    if args.max_samples is not None:
        image_ids = image_ids[: args.max_samples]

    rows = []

    for image_id in image_ids:
        labels = labels_by_image_id.get(image_id, set())

        if not labels:
            continue

        file_name = images_by_id[image_id]
        image_path = image_dir / file_name

        if not image_path.exists():
            continue

        row = {
            "frame_path": str(image_path),
            "source_dataset": "coco-2017",
            "source_split": split,
            "source_labels": ";".join(sorted(labels)),
        }

        for label in LABELS:
            row[label] = 1 if label in labels else 0

        rows.append(row)

    print("\nCategorias COCO mapeadas:")
    for label, count in sorted(mapped_category_counts.items()):
        print(f"- {label}: {count}")

    print("\nCategorias COCO ignoradas/não mapeadas:")
    for label, count in sorted(unmapped_category_counts.items())[:30]:
        print(f"- {label}: {count}")

    if not rows:
        raise ValueError(
            "Nenhuma imagem válida foi convertida do COCO manual. "
            "Isso geralmente significa que o mapeamento COCO_TO_SPECTRA_OBJECT não bateu com as labels atuais."
        )

    df = pd.DataFrame(rows)

    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nDataset Object criado a partir do COCO manual:")
    print(output_csv)
    print("Linhas salvas:", len(df))

    print_dataset_summary(df)

def map_folder_label_to_object(label_name: str) -> List[str]:
    normalized_folder = normalize_folder_name(label_name)
    normalized_coco = normalize_text(label_name)

    mapped_labels = []

    if normalized_folder in FOLDER_NAME_TO_SPECTRA_OBJECT:
        value = FOLDER_NAME_TO_SPECTRA_OBJECT[normalized_folder]
        if value is None:
            value = []
        elif isinstance(value, str):
            value = [value]
        mapped_labels.extend(value)

    if normalized_coco in COCO_TO_SPECTRA_OBJECT:
        value = COCO_TO_SPECTRA_OBJECT[normalized_coco]
        if value is None:
            value = []
        elif isinstance(value, str):
            value = [value]
        mapped_labels.extend(value)

    if normalized_folder in LABELS:
        mapped_labels.append(normalized_folder)

    return sorted({
        label
        for label in mapped_labels
        if label in LABELS
    })


def ensure_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "frame_path" not in df.columns:
        raise ValueError("Dataset precisa ter coluna frame_path.")

    df["frame_path"] = df["frame_path"].astype("string")
    df["frame_path"] = df["frame_path"].str.replace("\\", "/", regex=False)

    for label in LABELS:
        if label not in df.columns:
            df[label] = 0

        df[label] = (
            pd.to_numeric(df[label], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

    metadata_columns = [
        column
        for column in ["frame_path", "source_dataset", "source_split", "source_labels"]
        if column in df.columns
    ]

    return df[metadata_columns + LABELS]


def clean_dataset(df: pd.DataFrame, check_images: bool = True) -> pd.DataFrame:
    df = ensure_object_columns(df)
    df = df.dropna(subset=["frame_path"])
    df = df[df["frame_path"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["frame_path"])

    if check_images:
        before = len(df)
        df = df[df["frame_path"].apply(lambda value: Path(str(value)).exists())]
        removed = before - len(df)
        if removed:
            print(f"Imagens inexistentes removidas: {removed}")

    positive_count = df[LABELS].sum(axis=1)
    df = df[positive_count > 0]

    return df.reset_index(drop=True)


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\nResumo do dataset Object")
    print("Linhas:", len(df))
    print("Labels:", len(LABELS))

    label_counts = df[LABELS].sum().sort_values(ascending=False)
    non_zero = label_counts[label_counts > 0]
    zero = label_counts[label_counts == 0]

    print("\nLabels preenchidas:")
    for label, count in non_zero.items():
        print(f"- {label}: {int(count)}")

    if len(zero) > 0:
        print("\nLabels zeradas:")
        print(list(zero.index))
    else:
        print("\nNenhuma label zerada.")


def collect_images(input_dir: Path) -> List[Path]:
    image_paths = []

    for extension in IMAGE_EXTENSIONS:
        image_paths.extend(input_dir.rglob(f"*{extension}"))
        image_paths.extend(input_dir.rglob(f"*{extension.upper()}"))

    return sorted(set(image_paths))


def get_detection_labels_from_sample(sample: Any, field_name: str) -> List[str]:
    field_value = sample.get_field(field_name)

    if field_value is None:
        return []

    detections = getattr(field_value, "detections", None)

    if detections is None:
        return []

    return [detection.label for detection in detections if getattr(detection, "label", None)]


def command_from_coco(args: argparse.Namespace) -> None:
    try:
        import fiftyone.zoo as foz
    except ImportError as error:
        raise ImportError(
            "FiftyOne não está instalado. Instale com: pip install fiftyone"
        ) from error

    output_csv = Path(args.output_csv)

    coco_classes = [
        coco_label
        for coco_label, spectra_labels in COCO_TO_SPECTRA_OBJECT.items()
        if spectra_labels
    ]

    print("Carregando COCO 2017 pelo FiftyOne...")
    print("Split:", args.split)
    print("Max samples:", args.max_samples)

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=args.split,
        label_types=["detections"],
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        dataset_name=f"spectra-object-coco-{args.split}",
    )

    rows = []

    for sample in dataset:
        source_labels = get_detection_labels_from_sample(sample, args.detections_field)

        spectra_labels: Set[str] = set()

        for source_label in source_labels:
            spectra_labels.update(map_coco_label_to_object(source_label))

        if not spectra_labels:
            continue

        row = {
            "frame_path": normalize_path_text(sample.filepath),
            "source_dataset": "coco-2017",
            "source_split": args.split,
            "source_labels": ";".join(sorted(set(source_labels))),
        }

        for label in LABELS:
            row[label] = 1 if label in spectra_labels else 0

        rows.append(row)

    if not rows:
        raise ValueError(
            "Nenhuma imagem com labels mapeadas foi gerada. Confira o split/classes do COCO."
        )

    df = pd.DataFrame(rows)
    df = clean_dataset(df, check_images=not args.no_check_images)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nDataset COCO convertido para Object:")
    print(output_csv)
    print_dataset_summary(df)


def command_from_folders(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {input_dir}")

    image_paths = collect_images(input_dir)

    if args.max_rows is not None:
        image_paths = image_paths[: args.max_rows]

    rows = []
    skipped = 0

    for image_path in image_paths:
        relative_path = image_path.relative_to(input_dir)

        if len(relative_path.parts) < 2:
            skipped += 1
            continue

        source_label = relative_path.parts[0]
        spectra_labels = map_folder_label_to_object(source_label)

        if not spectra_labels:
            skipped += 1
            continue

        row = {
            "frame_path": normalize_path_text(image_path),
            "source_dataset": "object_folders",
            "source_split": "folders",
            "source_labels": source_label,
        }

        for label in LABELS:
            row[label] = 1 if label in spectra_labels else 0

        rows.append(row)

    if not rows:
        raise ValueError(
            "Nenhuma imagem foi convertida. A estrutura esperada é input_dir/label/imagem.jpg."
        )

    df = pd.DataFrame(rows)
    df = clean_dataset(df, check_images=not args.no_check_images)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nDataset Object criado a partir de pastas:")
    print(output_csv)
    print("Imagens encontradas:", len(image_paths))
    print("Ignoradas:", skipped)
    print_dataset_summary(df)


def command_clean(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV não encontrado: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    df = clean_dataset(df, check_images=not args.no_check_images)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nDataset limpo salvo em:")
    print(output_csv)
    print_dataset_summary(df)


def command_merge(args: argparse.Namespace) -> None:
    dataframes = []

    for input_path in args.inputs:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"CSV não encontrado: {input_path}")

        df = pd.read_csv(input_path, low_memory=False)
        df = ensure_object_columns(df)
        dataframes.append(df)

    merged = pd.concat(dataframes, ignore_index=True)
    merged = clean_dataset(merged, check_images=not args.no_check_images)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("\nDataset Object mesclado salvo em:")
    print(output_csv)
    print_dataset_summary(merged)


def command_validate(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df = clean_dataset(df, check_images=not args.no_check_images)
    print_dataset_summary(df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Constrói datasets para ai.spectra.Object."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    coco_parser = subparsers.add_parser("from-coco")
    coco_parser.add_argument("--split", default="validation", choices=["train", "validation"])
    coco_parser.add_argument("--dataset-dir", default="data/external/object/coco-2017")
    coco_parser.add_argument("--output-csv", default="data/datasets/Object/object_coco_labels.csv")
    coco_parser.add_argument("--max-samples", type=int, default=None)
    coco_parser.add_argument("--detections-field", default="ground_truth")
    coco_parser.add_argument("--seed", type=int, default=42)
    coco_parser.add_argument("--shuffle", action="store_true")
    coco_parser.add_argument("--no-check-images", action="store_true")
    coco_parser.set_defaults(func=command_from_coco)

    folders_parser = subparsers.add_parser("from-folders")
    folders_parser.add_argument("--input-dir", required=True)
    folders_parser.add_argument("--output-csv", default="data/datasets/Object/object_folder_labels.csv")
    folders_parser.add_argument("--max-rows", type=int, default=None)
    folders_parser.add_argument("--no-check-images", action="store_true")
    folders_parser.set_defaults(func=command_from_folders)

    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--input-csv", required=True)
    clean_parser.add_argument("--output-csv", required=True)
    clean_parser.add_argument("--no-check-images", action="store_true")
    clean_parser.set_defaults(func=command_clean)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", nargs="+", required=True)
    merge_parser.add_argument("--output-csv", required=True)
    merge_parser.add_argument("--no-check-images", action="store_true")
    merge_parser.set_defaults(func=command_merge)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--csv", required=True)
    validate_parser.add_argument("--no-check-images", action="store_true")
    validate_parser.set_defaults(func=command_validate)

    manual_coco_parser = subparsers.add_parser("from-manual-coco")
    manual_coco_parser.add_argument(
        "--coco-root",
        default="data/external/object/coco-2017",
    )
    manual_coco_parser.add_argument(
        "--split",
        choices=["train", "validation", "val"],
        default="validation",
    )
    manual_coco_parser.add_argument(
        "--output-csv",
        default="data/datasets/Object/object_coco_manual.csv",
    )
    manual_coco_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    manual_coco_parser.add_argument(
        "--no-check-images",
        action="store_true",
    )
    manual_coco_parser.set_defaults(func=command_from_manual_coco)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()