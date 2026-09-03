import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from PIL import Image

from ai.spectra.Atmosphere.labels import LABELS


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


BDD_TIME_OF_DAY_MAP = {
    "daytime": [
        "day",
        "bright",
    ],
    "night": [
        "night",
        "dark",
        "low_light",
    ],
    "dawn/dusk": [
        "dawn_dusk",
        "low_light",
    ],
    "dawn": [
        "dawn_dusk",
        "low_light",
    ],
    "dusk": [
        "dawn_dusk",
        "low_light",
    ],
}


BDD_WEATHER_MAP = {
    "clear": [
        "clear_weather",
    ],
    "overcast": [
        "cloudy",
    ],
    "partly cloudy": [
        "cloudy",
    ],
    "rainy": [
        "rainy",
        "cloudy",
    ],
    "snowy": [
        "snowy",
        "cloudy",
    ],
    "foggy": [
        "foggy",
        "cloudy",
        "low_light",
    ],
}


EXDARK_LABELS = [
    "dark",
    "low_light",
]


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = value.replace("\\", "/")
    value = value.replace("_", " ")
    value = " ".join(value.split())

    return value


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()

        return True

    except Exception:
        return False


def deterministic_split(
    value: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    number = int(digest[:8], 16) / 0xFFFFFFFF

    if number < train_ratio:
        return "train"

    if number < train_ratio + validation_ratio:
        return "validation"

    return "test"


def build_empty_row(frame_path: Path) -> Dict[str, object]:
    row = {
        "frame_path": str(frame_path),
    }

    for label in LABELS:
        row[label] = 0

    return row


def build_row(
    frame_path: Path,
    labels: Sequence[str],
    source_dataset: str,
    source_category: str,
    source_split: Optional[str] = None,
) -> Dict[str, object]:
    row = build_empty_row(frame_path)

    active_label_set = set(LABELS)

    for label in labels:
        if label in active_label_set:
            row[label] = 1

    row["source_dataset"] = source_dataset
    row["source_category"] = source_category
    row["source_split"] = source_split or ""
    row["generated_split"] = deterministic_split(str(frame_path))
    row["primary_label"] = infer_primary_label(row)

    return row


def infer_primary_label(row: Dict[str, object]) -> str:
    priority = [
        "night",
        "dawn_dusk",
        "day",
        "rainy",
        "snowy",
        "foggy",
        "cloudy",
        "clear_weather",
        "dark",
        "low_light",
        "bright",
        "backlit",
        "warm_light",
        "cold_light",
    ]

    for label in priority:
        if label in row and row[label] == 1:
            return label

    active_labels = [
        label
        for label in LABELS
        if row.get(label, 0) == 1
    ]

    if active_labels:
        return active_labels[0]

    return "unknown"


def find_bdd_label_files(bdd_root: Path) -> List[Path]:
    candidates = []

    for path in bdd_root.rglob("*.json"):
        name = path.name.lower()

        if "labels" not in name:
            continue

        if "image" not in name and "images" not in name:
            continue

        candidates.append(path)

    return sorted(candidates)


def find_bdd_images_root(bdd_root: Path) -> Optional[Path]:
    candidates = []

    for path in bdd_root.rglob("100k"):
        if not path.is_dir():
            continue

        train_dir = path / "train"
        val_dir = path / "val"
        test_dir = path / "test"

        if train_dir.exists() or val_dir.exists() or test_dir.exists():
            candidates.append(path)

    if candidates:
        return sorted(candidates, key=lambda item: len(str(item)))[0]

    image_candidates = []

    for path in bdd_root.rglob("images"):
        if path.is_dir():
            image_candidates.append(path)

    for image_root in sorted(image_candidates, key=lambda item: len(str(item))):
        possible_100k = image_root / "100k"

        if possible_100k.exists():
            return possible_100k

    return None


def load_bdd_json(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["frames", "images", "annotations", "data"]:
            value = data.get(key)

            if isinstance(value, list):
                return value

    raise ValueError(f"Formato JSON não reconhecido: {path}")


def resolve_bdd_image_path(
    images_root: Path,
    image_name: str,
    preferred_split: Optional[str],
) -> Optional[Path]:
    image_name = Path(image_name).name

    split_candidates = []

    if preferred_split:
        split_candidates.append(preferred_split)

    split_candidates.extend(
        [
            "train",
            "val",
            "test",
        ]
    )

    for split in split_candidates:
        candidate = images_root / split / image_name

        if candidate.exists():
            return candidate

    matches = list(images_root.rglob(image_name))

    if matches:
        return matches[0]

    return None


def labels_from_bdd_attributes(attributes: Dict[str, object]) -> List[str]:
    labels = set()

    time_of_day = normalize_text(attributes.get("timeofday"))
    weather = normalize_text(attributes.get("weather"))

    if time_of_day in BDD_TIME_OF_DAY_MAP:
        labels.update(BDD_TIME_OF_DAY_MAP[time_of_day])

    if weather in BDD_WEATHER_MAP:
        labels.update(BDD_WEATHER_MAP[weather])

    return sorted(labels)


def infer_bdd_split_from_label_file(path: Path) -> Optional[str]:
    name = path.name.lower()

    if "train" in name:
        return "train"

    if "val" in name or "validation" in name:
        return "val"

    if "test" in name:
        return "test"

    return None


def collect_rows_from_bdd100k(
    bdd_root: Path,
    max_samples: Optional[int] = None,
    validate_images: bool = False,
) -> List[Dict[str, object]]:
    bdd_root = Path(bdd_root)

    if not bdd_root.exists():
        raise FileNotFoundError(f"Pasta BDD100K não encontrada: {bdd_root}")

    label_files = find_bdd_label_files(bdd_root)
    images_root = find_bdd_images_root(bdd_root)

    if not label_files:
        raise FileNotFoundError(
            "Nenhum arquivo de labels do BDD100K foi encontrado dentro de: "
            f"{bdd_root}"
        )

    if images_root is None:
        raise FileNotFoundError(
            "Nenhuma pasta images/100k do BDD100K foi encontrada dentro de: "
            f"{bdd_root}"
        )

    print("BDD100K labels encontrados:")
    for path in label_files:
        print("-", path)

    print("BDD100K images root:", images_root)

    rows = []

    for label_file in label_files:
        source_split = infer_bdd_split_from_label_file(label_file)
        items = load_bdd_json(label_file)

        for item in items:
            image_name = item.get("name") or item.get("file_name") or item.get("image")

            if not image_name:
                continue

            attributes = item.get("attributes", {})

            if not isinstance(attributes, dict):
                continue

            labels = labels_from_bdd_attributes(attributes)

            if not labels:
                continue

            image_path = resolve_bdd_image_path(
                images_root=images_root,
                image_name=image_name,
                preferred_split=source_split,
            )

            if image_path is None:
                continue

            if validate_images and not is_valid_image(image_path):
                continue

            time_of_day = normalize_text(attributes.get("timeofday"))
            weather = normalize_text(attributes.get("weather"))

            source_category = f"timeofday={time_of_day};weather={weather}"

            row = build_row(
                frame_path=image_path,
                labels=labels,
                source_dataset="bdd100k",
                source_category=source_category,
                source_split=source_split,
            )

            rows.append(row)

            if max_samples is not None and len(rows) >= max_samples:
                return rows

    return rows


def collect_rows_from_exdark(
    exdark_root: Path,
    max_samples: Optional[int] = None,
    validate_images: bool = False,
) -> List[Dict[str, object]]:
    exdark_root = Path(exdark_root)

    if not exdark_root.exists():
        raise FileNotFoundError(f"Pasta ExDark não encontrada: {exdark_root}")

    image_paths = [
        path
        for path in exdark_root.rglob("*")
        if is_image_file(path)
    ]

    image_paths = sorted(image_paths)

    rows = []

    for image_path in image_paths:
        if validate_images and not is_valid_image(image_path):
            continue

        try:
            relative_path = image_path.relative_to(exdark_root)
            source_category = str(relative_path.parent).replace("\\", "/")
        except Exception:
            source_category = image_path.parent.name

        row = build_row(
            frame_path=image_path,
            labels=EXDARK_LABELS,
            source_dataset="exdark",
            source_category=source_category,
            source_split="",
        )

        rows.append(row)

        if max_samples is not None and len(rows) >= max_samples:
            break

    return rows


def dedupe_rows_by_frame_path(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_path = {}

    for row in rows:
        frame_path = row["frame_path"]

        if frame_path not in by_path:
            by_path[frame_path] = row
            continue

        existing = by_path[frame_path]

        for label in LABELS:
            existing[label] = int(existing.get(label, 0)) or int(row.get(label, 0))

        existing["primary_label"] = infer_primary_label(existing)

    return list(by_path.values())


def balance_rows(
    rows: List[Dict[str, object]],
    max_per_primary_label: Optional[int] = None,
    max_per_label: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, object]]:
    if max_per_primary_label is None and max_per_label is None:
        return rows

    random.seed(seed)

    shuffled_rows = rows[:]
    random.shuffle(shuffled_rows)

    selected_rows = []
    primary_counts = defaultdict(int)
    label_counts = defaultdict(int)

    for row in shuffled_rows:
        primary_label = str(row.get("primary_label", "unknown"))

        if max_per_primary_label is not None:
            if primary_counts[primary_label] >= max_per_primary_label:
                continue

        if max_per_label is not None:
            active_labels = [
                label
                for label in LABELS
                if int(row.get(label, 0)) == 1
            ]

            would_exceed_all = True

            for label in active_labels:
                if label_counts[label] < max_per_label:
                    would_exceed_all = False
                    break

            if would_exceed_all:
                continue

        selected_rows.append(row)
        primary_counts[primary_label] += 1

        for label in LABELS:
            if int(row.get(label, 0)) == 1:
                label_counts[label] += 1

    selected_rows.sort(
        key=lambda item: str(item["frame_path"])
    )

    return selected_rows


def create_report(rows: List[Dict[str, object]]) -> Dict[str, object]:
    report = {
        "total_rows": len(rows),
        "labels": {},
        "primary_labels": {},
        "source_datasets": {},
        "generated_splits": {},
        "source_categories_top": {},
    }

    for label in LABELS:
        report["labels"][label] = int(
            sum(
                int(row.get(label, 0))
                for row in rows
            )
        )

    for row in rows:
        primary_label = str(row.get("primary_label", "unknown"))
        source_dataset = str(row.get("source_dataset", "unknown"))
        generated_split = str(row.get("generated_split", "unknown"))
        source_category = str(row.get("source_category", "unknown"))

        report["primary_labels"][primary_label] = (
            report["primary_labels"].get(primary_label, 0) + 1
        )

        report["source_datasets"][source_dataset] = (
            report["source_datasets"].get(source_dataset, 0) + 1
        )

        report["generated_splits"][generated_split] = (
            report["generated_splits"].get(generated_split, 0) + 1
        )

        report["source_categories_top"][source_category] = (
            report["source_categories_top"].get(source_category, 0) + 1
        )

    report["labels"] = dict(
        sorted(
            report["labels"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    report["primary_labels"] = dict(
        sorted(
            report["primary_labels"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    report["source_datasets"] = dict(
        sorted(
            report["source_datasets"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    report["generated_splits"] = dict(
        sorted(
            report["generated_splits"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    report["source_categories_top"] = dict(
        sorted(
            report["source_categories_top"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:80]
    )

    return report


def save_rows(rows: List[Dict[str, object]], output_csv: Path):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "frame_path",
        *LABELS,
        "primary_label",
        "generated_split",
        "source_dataset",
        "source_category",
        "source_split",
    ]

    dataframe = pd.DataFrame(rows)

    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = ""

    dataframe = dataframe[columns]
    dataframe.to_csv(output_csv, index=False)

    return output_csv


def save_report(report: Dict[str, object], output_report: Path):
    output_report = Path(output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    with open(output_report, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    return output_report


def print_report(report: Dict[str, object]):
    print("\n" + "=" * 80)
    print("RELATÓRIO ATMOSPHERE DATASET")
    print("=" * 80)

    print("Total:", report["total_rows"])

    print("\nPor dataset:")
    for key, value in report["source_datasets"].items():
        print(f"- {key}: {value}")

    print("\nSplits gerados:")
    for key, value in report["generated_splits"].items():
        print(f"- {key}: {value}")

    print("\nLabels:")
    for key, value in report["labels"].items():
        print(f"- {key}: {value}")

    print("\nPrimary labels:")
    for key, value in report["primary_labels"].items():
        print(f"- {key}: {value}")

    print("=" * 80)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Builder de dataset para SpectraAtmosphereNet usando BDD100K e ExDark."
    )

    parser.add_argument(
        "--bdd-root",
        default="data/external/bdd100k",
        help="Pasta raiz do BDD100K.",
    )

    parser.add_argument(
        "--exdark-root",
        default="data/external/ExDark",
        help="Pasta raiz do ExDark.",
    )

    parser.add_argument(
        "--sources",
        default="bdd100k,exdark",
        help="Fontes separadas por vírgula: bdd100k,exdark.",
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/spectra_atmosphere_v1.csv",
        help="Caminho do CSV final.",
    )

    parser.add_argument(
        "--output-report",
        default="data/datasets/spectra_atmosphere_v1_report.json",
        help="Caminho do relatório JSON.",
    )

    parser.add_argument(
        "--max-bdd-samples",
        type=int,
        default=None,
        help="Limite máximo de imagens vindas do BDD100K.",
    )

    parser.add_argument(
        "--max-exdark-samples",
        type=int,
        default=None,
        help="Limite máximo de imagens vindas do ExDark.",
    )

    parser.add_argument(
        "--max-per-primary-label",
        type=int,
        default=None,
        help="Limite máximo por primary_label.",
    )

    parser.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Limite aproximado por label multilabel.",
    )

    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Valida se as imagens podem ser abertas. Mais lento.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    selected_sources = {
        source.strip().lower()
        for source in args.sources.split(",")
        if source.strip()
    }

    rows = []

    if "bdd100k" in selected_sources:
        print("=" * 80)
        print("Coletando BDD100K")
        print("=" * 80)

        bdd_rows = collect_rows_from_bdd100k(
            bdd_root=Path(args.bdd_root),
            max_samples=args.max_bdd_samples,
            validate_images=args.validate_images,
        )

        print("BDD100K rows:", len(bdd_rows))
        rows.extend(bdd_rows)

    if "exdark" in selected_sources:
        print("=" * 80)
        print("Coletando ExDark")
        print("=" * 80)

        exdark_rows = collect_rows_from_exdark(
            exdark_root=Path(args.exdark_root),
            max_samples=args.max_exdark_samples,
            validate_images=args.validate_images,
        )

        print("ExDark rows:", len(exdark_rows))
        rows.extend(exdark_rows)

    rows = dedupe_rows_by_frame_path(rows)

    rows = balance_rows(
        rows=rows,
        max_per_primary_label=args.max_per_primary_label,
        max_per_label=args.max_per_label,
        seed=args.seed,
    )

    report = create_report(rows)

    save_rows(
        rows=rows,
        output_csv=Path(args.output_csv),
    )

    save_report(
        report=report,
        output_report=Path(args.output_report),
    )

    print_report(report)

    print("\nCSV salvo em:", args.output_csv)
    print("Relatório salvo em:", args.output_report)


if __name__ == "__main__":
    main()