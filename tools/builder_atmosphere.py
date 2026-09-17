import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ai.spectra.Atmosphere.labels import LABELS


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


TRANSIENT_ATTRIBUTES = [
    "dirty",
    "daylight",
    "night",
    "sunrisesunset",
    "dawndusk",
    "sunny",
    "clouds",
    "fog",
    "storm",
    "snow",
    "warm",
    "cold",
    "busy",
    "beautiful",
    "flowers",
    "spring",
    "summer",
    "autumn",
    "winter",
    "glowing",
    "colorful",
    "dull",
    "rugged",
    "midday",
    "dark",
    "bright",
    "dry",
    "moist",
    "windy",
    "rain",
    "ice",
    "cluttered",
    "soothing",
    "stressful",
    "exciting",
    "sentimental",
    "mysterious",
    "boring",
    "gloomy",
    "lush",
]


BDD_TIME_MAP = {
    "daytime": "day",
    "night": "night",
    "dawn/dusk": "dawn_dusk",
}


BDD_WEATHER_MAP = {
    "clear": "clear_weather",
    "overcast": "cloudy",
    "partly cloudy": "cloudy",
    "foggy": "foggy",
    "rainy": "rainy",
    "snowy": "snowy",
}


ACDC_MAP = {
    "fog": "foggy",
    "night": "night",
    "rain": "rainy",
    "snow": "snowy",
}


def normalize_text(value):
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = value.replace("\\", "/")
    value = value.replace("_", " ")
    value = " ".join(value.split())

    return value


def is_image_file(path):
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def deterministic_split(
    value,
    train_ratio=0.80,
    validation_ratio=0.10,
):
    digest = hashlib.md5(
        str(value).encode("utf-8")
    ).hexdigest()

    number = int(
        digest[:8],
        16,
    ) / 0xFFFFFFFF

    if number < train_ratio:
        return "train"

    if number < train_ratio + validation_ratio:
        return "validation"

    return "test"


def normalize_split(split):
    split = normalize_text(split)

    if split == "train":
        return "train"

    if split in {
        "val",
        "validation",
    }:
        return "validation"

    if split == "test":
        return "test"

    return ""


def create_empty_row(frame_path):
    row = {
        "frame_path": str(frame_path),
    }

    for label in LABELS:
        row[label] = 0

    return row


def set_label(row, label, value=1):
    if label not in LABELS:
        return

    row[label] = value


def active_labels(row, threshold=0.5):
    return [
        label
        for label in LABELS
        if float(row.get(label, 0)) >= threshold
    ]


def infer_primary_label(row):
    candidates = []

    for label in LABELS:
        score = float(row.get(label, 0))

        if score > 0:
            candidates.append(
                (
                    label,
                    score,
                )
            )

    if not candidates:
        return "unknown"

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return candidates[0][0]


def finish_row(
    row,
    source_dataset,
    source_category,
    source_split="",
    generated_split=None,
):
    row["source_dataset"] = source_dataset
    row["source_category"] = source_category
    row["source_split"] = source_split

    if generated_split:
        row["generated_split"] = generated_split

    else:
        row["generated_split"] = deterministic_split(
            row["frame_path"]
        )

    row["primary_label"] = infer_primary_label(row)

    return row


def find_first_directory(
    root,
    directory_name,
):
    directory_name = directory_name.lower()

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        if path.name.lower() == directory_name:
            return path

    return None


def find_first_file(
    root,
    filename,
):
    filename = filename.lower()

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.name.lower() == filename:
            return path

    return None


# ============================================================
# ACDC
# ============================================================


def find_acdc_rgb_root(external_root):
    candidates = []

    for path in external_root.rglob("rgb_anon"):
        if not path.is_dir():
            continue

        expected = {
            "fog",
            "night",
            "rain",
            "snow",
        }

        children = {
            child.name.lower()
            for child in path.iterdir()
            if child.is_dir()
        }

        if expected.issubset(children):
            candidates.append(path)

    if not candidates:
        return None

    return candidates[0]


def collect_acdc_rows(
    external_root,
    max_per_condition=None,
):
    rgb_root = find_acdc_rgb_root(
        external_root
    )

    if rgb_root is None:
        print("AVISO: ACDC não encontrado.")
        return []

    print("ACDC encontrado em:", rgb_root)

    rows = []

    for condition, label in ACDC_MAP.items():
        condition_dir = rgb_root / condition

        if not condition_dir.exists():
            continue

        condition_rows = []

        for split_name in [
            "train",
            "val",
            "test",
        ]:
            split_dir = condition_dir / split_name

            if not split_dir.exists():
                continue

            generated_split = normalize_split(
                split_name
            )

            for image_path in split_dir.rglob("*"):
                if not is_image_file(image_path):
                    continue

                row = create_empty_row(
                    image_path
                )

                set_label(
                    row,
                    label,
                    1,
                )

                row = finish_row(
                    row=row,
                    source_dataset="acdc",
                    source_category=condition,
                    source_split=split_name,
                    generated_split=generated_split,
                )

                condition_rows.append(row)

        if max_per_condition is not None:
            random.Random(42).shuffle(
                condition_rows
            )

            condition_rows = condition_rows[
                :max_per_condition
            ]

        rows.extend(
            condition_rows
        )

    return rows


# ============================================================
# TRANSIENT ATTRIBUTES
# ============================================================


def discover_transient_paths(
    external_root,
):
    annotations_path = find_first_file(
        external_root,
        "annotations.tsv",
    )

    image_root = find_first_directory(
        external_root,
        "imageAlignedLD",
    )

    if image_root is None:
        image_root = find_first_directory(
            external_root,
            "imageLD",
        )

    holdout_root = find_first_directory(
        external_root,
        "holdout_split",
    )

    return (
        annotations_path,
        image_root,
        holdout_root,
    )


def load_transient_annotations(
    annotations_path,
):
    annotations = {}

    with annotations_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.reader(
            file,
            delimiter="\t",
        )

        for row in reader:
            if not row:
                continue

            key = (
                row[0]
                .strip()
                .replace("\\", "/")
            )

            values = []

            for column in row[1:]:
                column = column.strip()

                if not column:
                    continue

                value_text = (
                    column
                    .split(",")[0]
                )

                try:
                    values.append(
                        float(value_text)
                    )

                except ValueError:
                    values.append(0.0)

            if len(values) < 40:
                continue

            annotations[key] = values[:40]

    return annotations


def load_transient_split(
    holdout_root,
):
    split_by_path = {}

    if holdout_root is None:
        return split_by_path

    candidates = {
        "training.txt": "train",
        "train.txt": "train",
        "test.txt": "test",
    }

    for filename, split in candidates.items():
        path = holdout_root / filename

        if not path.exists():
            continue

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                relative_path = (
                    line.strip()
                    .replace("\\", "/")
                )

                if relative_path:
                    split_by_path[
                        relative_path
                    ] = split

    return split_by_path


def transient_values_to_row(
    frame_path,
    values,
    positive_threshold,
):
    row = create_empty_row(
        frame_path
    )

    values_by_attribute = {
        attribute: float(value)
        for attribute, value in zip(
            TRANSIENT_ATTRIBUTES,
            values,
        )
    }

    mappings = {
        "day": values_by_attribute[
            "daylight"
        ],

        "night": values_by_attribute[
            "night"
        ],

        "dawn_dusk": max(
            values_by_attribute[
                "sunrisesunset"
            ],
            values_by_attribute[
                "dawndusk"
            ],
        ),

        "sunny": values_by_attribute[
            "sunny"
        ],

        "bright": values_by_attribute[
            "bright"
        ],

        "cloudy": values_by_attribute[
            "clouds"
        ],

        "foggy": values_by_attribute[
            "fog"
        ],

        "rainy": values_by_attribute[
            "rain"
        ],

        "snowy": values_by_attribute[
            "snow"
        ],
    }

    for label, score in mappings.items():
        if label not in LABELS:
            continue

        row[label] = int(
            score >= positive_threshold
        )

    return row


def collect_transient_rows(
    external_root,
    positive_threshold=0.70,
    max_samples=None,
):
    (
        annotations_path,
        image_root,
        holdout_root,
    ) = discover_transient_paths(
        external_root
    )

    if (
        annotations_path is None
        or image_root is None
    ):
        print(
            "AVISO: Transient Attributes "
            "não encontrado."
        )

        return []

    print(
        "Transient annotations:",
        annotations_path,
    )

    print(
        "Transient images:",
        image_root,
    )

    print(
        "Transient holdout:",
        holdout_root,
    )

    annotations = (
        load_transient_annotations(
            annotations_path
        )
    )

    split_by_path = (
        load_transient_split(
            holdout_root
        )
    )

    rows = []

    for relative_path, values in annotations.items():
        normalized_path = (
            relative_path
            .replace("\\", "/")
        )

        image_path = (
            image_root
            / Path(normalized_path)
        )

        if not image_path.exists():
            continue

        row = transient_values_to_row(
            frame_path=image_path,
            values=values,
            positive_threshold=positive_threshold,
        )

        if not active_labels(row):
            continue

        source_split = (
            split_by_path
            .get(
                normalized_path,
                "",
            )
        )

        row = finish_row(
            row=row,
            source_dataset="transient_attributes",
            source_category="human_attributes",
            source_split=source_split,
            generated_split=(
                normalize_split(
                    source_split
                )
                if source_split
                else None
            ),
        )

        rows.append(row)

    random.Random(42).shuffle(rows)

    if max_samples is not None:
        rows = rows[:max_samples]

    return rows


# ============================================================
# SUN ATTRIBUTE DATABASE
# ============================================================


def load_mat_variable(
    path,
    preferred_names,
):
    try:
        from scipy.io import loadmat

    except ImportError as exc:
        raise ImportError(
            "O SUN Attribute Database "
            "precisa do scipy.\n"
            "Instale com: pip install scipy"
        ) from exc

    data = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
    )

    for name in preferred_names:
        if name in data:
            return data[name]

    candidates = [
        value
        for key, value in data.items()
        if not key.startswith("__")
    ]

    if not candidates:
        raise ValueError(
            f"Nenhuma variável encontrada em {path}"
        )

    return candidates[0]


def matlab_to_string(value):
    current = value

    while (
        isinstance(current, np.ndarray)
        and current.size == 1
    ):
        current = current.item()

    if isinstance(current, bytes):
        return current.decode(
            "utf-8",
            errors="ignore",
        )

    if isinstance(current, str):
        return current

    if isinstance(current, np.ndarray):
        flat = current.reshape(-1)

        if all(
            isinstance(item, str)
            for item in flat
        ):
            return "".join(flat)

        if all(
            isinstance(item, bytes)
            for item in flat
        ):
            return "".join(
                item.decode(
                    "utf-8",
                    errors="ignore",
                )
                for item in flat
            )

    return str(current)


def matlab_string_list(value):
    array = np.asarray(
        value,
        dtype=object,
    ).reshape(-1)

    return [
        matlab_to_string(item)
        .strip()
        .replace("\\", "/")
        for item in array
    ]


def discover_sun_files(
    external_root,
):
    labels_path = find_first_file(
        external_root,
        "attributeLabels_continuous.mat",
    )

    images_mat_path = find_first_file(
        external_root,
        "images.mat",
    )

    attributes_path = find_first_file(
        external_root,
        "attributes.mat",
    )

    return (
        labels_path,
        images_mat_path,
        attributes_path,
    )


def find_sun_image_root(
    external_root,
    image_names,
):
    candidates = [
        path
        for path in external_root.rglob("images")
        if path.is_dir()
    ]

    best_root = None
    best_hits = -1

    probes = image_names[:30]

    for candidate in candidates:
        hits = 0

        for image_name in probes:
            relative_name = (
                image_name
                .lstrip("/")
            )

            if (
                candidate
                / Path(relative_name)
            ).exists():
                hits += 1

        if hits > best_hits:
            best_hits = hits
            best_root = candidate

    if best_hits <= 0:
        return None

    return best_root


def find_attribute_index(
    attributes,
    possible_names,
):
    normalized = {
        normalize_text(name): index
        for index, name
        in enumerate(attributes)
    }

    for name in possible_names:
        key = normalize_text(name)

        if key in normalized:
            return normalized[key]

    return None


def collect_sun_rows(
    external_root,
    positive_threshold=0.75,
    max_samples=None,
):
    (
        labels_path,
        images_mat_path,
        attributes_path,
    ) = discover_sun_files(
        external_root
    )

    if (
        labels_path is None
        or images_mat_path is None
        or attributes_path is None
    ):
        print(
            "AVISO: SUN Attribute Database "
            "não encontrado."
        )

        return []

    image_names = matlab_string_list(
        load_mat_variable(
            images_mat_path,
            [
                "images",
            ],
        )
    )

    attributes = matlab_string_list(
        load_mat_variable(
            attributes_path,
            [
                "attributes",
            ],
        )
    )

    label_matrix = np.asarray(
        load_mat_variable(
            labels_path,
            [
                "attributeLabels_continuous",
                "attribute_labels",
                "labels",
            ],
        ),
        dtype=np.float32,
    )

    label_matrix = np.squeeze(
        label_matrix
    )

    if (
        label_matrix.shape[0]
        != len(image_names)
        and label_matrix.shape[1]
        == len(image_names)
    ):
        label_matrix = (
            label_matrix.T
        )

    if (
        label_matrix.shape[0]
        != len(image_names)
    ):
        raise ValueError(
            "Formato inesperado no SUN. "
            f"Imagens: {len(image_names)} | "
            f"labels: {label_matrix.shape}"
        )

    image_root = find_sun_image_root(
        external_root=external_root,
        image_names=image_names,
    )

    if image_root is None:
        print(
            "AVISO: imagens SUN não encontradas."
        )

        return []

    print(
        "SUN metadata:",
        labels_path.parent,
    )

    print(
        "SUN images:",
        image_root,
    )

    sunny_index = find_attribute_index(
        attributes,
        [
            "direct sun/sunny",
            "sunny",
        ],
    )

    clouds_index = find_attribute_index(
        attributes,
        [
            "clouds",
        ],
    )

    usable_indices = {
        "sunny": sunny_index,
        "cloudy": clouds_index,
    }

    rows = []

    for index, image_name in enumerate(
        image_names
    ):
        relative_name = (
            image_name
            .lstrip("/")
        )

        image_path = (
            image_root
            / Path(relative_name)
        )

        if not image_path.exists():
            continue

        row = create_empty_row(
            image_path
        )

        for label, attribute_index in (
            usable_indices.items()
        ):
            if (
                label not in LABELS
                or attribute_index is None
            ):
                continue

            score = float(
                label_matrix[
                    index,
                    attribute_index,
                ]
            )

            row[label] = int(
                score
                >= positive_threshold
            )

        if not active_labels(row):
            continue

        row = finish_row(
            row=row,
            source_dataset="sun_attributes",
            source_category="human_attributes",
            source_split="",
            generated_split=None,
        )

        rows.append(row)

    random.Random(42).shuffle(rows)

    if max_samples is not None:
        rows = rows[:max_samples]

    return rows


# ============================================================
# BDD100K
# ============================================================


def find_bdd_label_files(
    external_root,
):
    candidates = []

    for path in external_root.rglob(
        "*.json"
    ):
        name = path.name.lower()

        if "bdd100k" not in name:
            continue

        if "labels" not in name:
            continue

        if "images" not in name:
            continue

        candidates.append(path)

    return sorted(candidates)


def find_bdd_images_root(
    external_root,
):
    for path in external_root.rglob(
        "100k"
    ):
        if not path.is_dir():
            continue

        if (
            (path / "train").exists()
            or (path / "val").exists()
            or (path / "test").exists()
        ):
            return path

    return None


def infer_bdd_split(
    path,
):
    name = path.name.lower()

    if "train" in name:
        return "train"

    if "val" in name:
        return "validation"

    if "test" in name:
        return "test"

    return ""


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_bdd_image(
    image_root,
    image_name,
    source_split,
):
    image_name = Path(
        image_name
    ).name

    split_candidates = []

    if source_split == "train":
        split_candidates.append(
            "train"
        )

    elif source_split == "validation":
        split_candidates.append(
            "val"
        )

    elif source_split == "test":
        split_candidates.append(
            "test"
        )

    split_candidates.extend(
        [
            "train",
            "val",
            "test",
        ]
    )

    for split in split_candidates:
        candidate = (
            image_root
            / split
            / image_name
        )

        if candidate.exists():
            return candidate

    return None


def collect_bdd100k_rows(
    external_root,
    max_samples=None,
):
    label_files = (
        find_bdd_label_files(
            external_root
        )
    )

    image_root = (
        find_bdd_images_root(
            external_root
        )
    )

    if (
        not label_files
        or image_root is None
    ):
        print(
            "AVISO: BDD100K não encontrado."
        )

        return []

    print(
        "BDD images:",
        image_root,
    )

    rows = []

    for label_file in label_files:
        source_split = infer_bdd_split(
            label_file
        )

        data = load_json(
            label_file
        )

        if isinstance(data, dict):
            data = (
                data.get("frames")
                or data.get("images")
                or data.get("data")
                or []
            )

        for item in data:
            image_name = (
                item.get("name")
                or item.get("file_name")
                or item.get("image")
            )

            attributes = item.get(
                "attributes",
                {},
            )

            if (
                not image_name
                or not isinstance(
                    attributes,
                    dict,
                )
            ):
                continue

            row = None

            time_of_day = normalize_text(
                attributes.get(
                    "timeofday"
                )
            )

            weather = normalize_text(
                attributes.get(
                    "weather"
                )
            )

            time_label = (
                BDD_TIME_MAP.get(
                    time_of_day
                )
            )

            weather_label = (
                BDD_WEATHER_MAP.get(
                    weather
                )
            )

            if (
                time_label not in LABELS
                and weather_label not in LABELS
            ):
                continue

            image_path = resolve_bdd_image(
                image_root=image_root,
                image_name=image_name,
                source_split=source_split,
            )

            if image_path is None:
                continue

            row = create_empty_row(
                image_path
            )

            if time_label:
                set_label(
                    row,
                    time_label,
                    1,
                )

            if weather_label:
                set_label(
                    row,
                    weather_label,
                    1,
                )

            if not active_labels(row):
                continue

            row = finish_row(
                row=row,
                source_dataset="bdd100k",
                source_category=(
                    f"time={time_of_day};"
                    f"weather={weather}"
                ),
                source_split=source_split,
                generated_split=source_split or None,
            )

            rows.append(row)

            if (
                max_samples is not None
                and len(rows)
                >= max_samples
            ):
                return rows

    return rows


# ============================================================
# EXDARK
# ============================================================


def find_exdark_root(
    external_root,
):
    for path in external_root.rglob("*"):
        if (
            path.is_dir()
            and path.name.lower()
            == "exdark"
        ):
            return path

    return None


def collect_exdark_rows(
    external_root,
    max_samples=None,
    seed=42,
):
    root = find_exdark_root(
        external_root
    )

    if root is None:
        print(
            "AVISO: ExDark não encontrado."
        )

        return []

    print(
        "ExDark encontrado em:",
        root,
    )

    paths = [
        path
        for path in root.rglob("*")
        if is_image_file(path)
    ]

    random.Random(seed).shuffle(
        paths
    )

    if max_samples is not None:
        paths = paths[
            :max_samples
        ]

    rows = []

    for image_path in paths:
        row = create_empty_row(
            image_path
        )

        if "low_light" in LABELS:
            set_label(
                row,
                "low_light",
                1,
            )

        elif "dark" in LABELS:
            set_label(
                row,
                "dark",
                1,
            )

        else:
            continue

        row = finish_row(
            row=row,
            source_dataset="exdark",
            source_category=(
                str(
                    image_path.parent
                    .relative_to(root)
                )
                .replace("\\", "/")
            ),
        )

        rows.append(row)

    return rows


# ============================================================
# FINAL DATASET
# ============================================================


def dedupe_rows(rows):
    unique = {}

    for row in rows:
        key = str(
            Path(
                row["frame_path"]
            ).resolve()
        ).lower()

        if key not in unique:
            unique[key] = row
            continue

        existing = unique[key]

        for label in LABELS:
            existing[label] = max(
                float(
                    existing.get(
                        label,
                        0,
                    )
                ),
                float(
                    row.get(
                        label,
                        0,
                    )
                ),
            )

        existing[
            "primary_label"
        ] = infer_primary_label(
            existing
        )

    return list(
        unique.values()
    )


def cap_per_label(
    rows,
    max_per_label=None,
    seed=42,
):
    if max_per_label is None:
        return rows

    rng = random.Random(
        seed
    )

    shuffled = rows[:]
    rng.shuffle(shuffled)

    counts = defaultdict(int)
    selected = []

    for row in shuffled:
        labels = active_labels(
            row
        )

        if not labels:
            continue

        useful = any(
            counts[label]
            < max_per_label
            for label in labels
        )

        if not useful:
            continue

        selected.append(row)

        for label in labels:
            counts[label] += 1

    return selected


def create_report(rows):
    label_counts = {}

    for label in LABELS:
        label_counts[label] = sum(
            1
            for row in rows
            if float(
                row.get(
                    label,
                    0,
                )
            ) >= 0.5
        )

    source_counts = defaultdict(int)
    split_counts = defaultdict(int)
    primary_counts = defaultdict(int)

    for row in rows:
        source_counts[
            row.get(
                "source_dataset",
                "unknown",
            )
        ] += 1

        split_counts[
            row.get(
                "generated_split",
                "unknown",
            )
        ] += 1

        primary_counts[
            row.get(
                "primary_label",
                "unknown",
            )
        ] += 1

    return {
        "total_rows": len(rows),
        "labels": dict(
            sorted(
                label_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "sources": dict(
            sorted(
                source_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "splits": dict(
            sorted(
                split_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "primary_labels": dict(
            sorted(
                primary_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    }


def save_dataset(
    rows,
    output_csv,
):
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "frame_path",
        *LABELS,
        "primary_label",
        "generated_split",
        "source_dataset",
        "source_category",
        "source_split",
    ]

    dataframe = pd.DataFrame(
        rows
    )

    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = ""

    dataframe = dataframe[
        columns
    ]

    dataframe.to_csv(
        output_csv,
        index=False,
    )


def save_report(
    report,
    output_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )


def print_report(
    report,
):
    print(
        "\n"
        + "=" * 80
    )

    print(
        "SPECTRA ATMOSPHERE DATASET"
    )

    print(
        "=" * 80
    )

    print(
        "Total:",
        report["total_rows"],
    )

    print(
        "\nFONTES"
    )

    for name, count in (
        report["sources"]
        .items()
    ):
        print(
            f"{name}: {count}"
        )

    print(
        "\nSPLITS"
    )

    for name, count in (
        report["splits"]
        .items()
    ):
        print(
            f"{name}: {count}"
        )

    print(
        "\nLABELS"
    )

    for name, count in (
        report["labels"]
        .items()
    ):
        print(
            f"{name}: {count}"
        )

    print(
        "=" * 80
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Gera o dataset do SpectraAtmosphereNet "
            "a partir de data/external."
        )
    )

    parser.add_argument(
        "--external-root",
        default="data/external",
    )

    parser.add_argument(
        "--sources",
        default=(
            "acdc,"
            "transient,"
            "sun,"
            "exdark,"
            "bdd100k"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "data/datasets/"
            "spectra_atmosphere_v2.csv"
        ),
    )

    parser.add_argument(
        "--output-report",
        default=(
            "data/datasets/"
            "spectra_atmosphere_v2_report.json"
        ),
    )

    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.70,
        help=(
            "Threshold para transformar "
            "labels humanas contínuas "
            "de Transient/SUN em positivas."
        ),
    )

    parser.add_argument(
        "--max-acdc-per-condition",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-transient-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-sun-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-exdark-samples",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--max-bdd-samples",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--max-per-label",
        type=int,
        default=None,
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

    external_root = Path(
        args.external_root
    )

    if not external_root.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: "
            f"{external_root}"
        )

    sources = {
        source.strip().lower()
        for source
        in args.sources.split(",")
        if source.strip()
    }

    rows = []

    if "acdc" in sources:
        print(
            "\nColetando ACDC..."
        )

        new_rows = collect_acdc_rows(
            external_root=external_root,
            max_per_condition=args.max_acdc_per_condition,
        )

        print(
            "ACDC:",
            len(new_rows),
        )

        rows.extend(
            new_rows
        )

    if "transient" in sources:
        print(
            "\nColetando Transient Attributes..."
        )

        new_rows = collect_transient_rows(
            external_root=external_root,
            positive_threshold=(
                args.positive_threshold
            ),
            max_samples=(
                args.max_transient_samples
            ),
        )

        print(
            "Transient:",
            len(new_rows),
        )

        rows.extend(
            new_rows
        )

    if "sun" in sources:
        print(
            "\nColetando SUN Attributes..."
        )

        new_rows = collect_sun_rows(
            external_root=external_root,
            positive_threshold=(
                args.positive_threshold
            ),
            max_samples=args.max_sun_samples,
        )

        print(
            "SUN:",
            len(new_rows),
        )

        rows.extend(
            new_rows
        )

    if "exdark" in sources:
        print(
            "\nColetando ExDark..."
        )

        new_rows = collect_exdark_rows(
            external_root=external_root,
            max_samples=(
                args.max_exdark_samples
            ),
            seed=args.seed,
        )

        print(
            "ExDark:",
            len(new_rows),
        )

        rows.extend(
            new_rows
        )

    if "bdd100k" in sources:
        print(
            "\nColetando BDD100K..."
        )

        new_rows = collect_bdd100k_rows(
            external_root=external_root,
            max_samples=(
                args.max_bdd_samples
            ),
        )

        print(
            "BDD100K:",
            len(new_rows),
        )

        rows.extend(
            new_rows
        )

    rows = dedupe_rows(
        rows
    )

    rows = cap_per_label(
        rows=rows,
        max_per_label=(
            args.max_per_label
        ),
        seed=args.seed,
    )

    random.Random(
        args.seed
    ).shuffle(
        rows
    )

    report = create_report(
        rows
    )

    output_csv = Path(
        args.output_csv
    )

    output_report = Path(
        args.output_report
    )

    save_dataset(
        rows=rows,
        output_csv=output_csv,
    )

    save_report(
        report=report,
        output_path=output_report,
    )

    print_report(
        report
    )

    print(
        "\nCSV:",
        output_csv,
    )

    print(
        "Relatório:",
        output_report,
    )


if __name__ == "__main__":
    main()