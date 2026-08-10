import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ai.spectra.Scene.labels import SPECTRA_SCENE_LABELS


"""
Builder otimizado para SpectraSceneNet v2 usando datasets de cenas/lugares.

Objetivo:
- Priorizar reconhecimento de locais: rua, deserto, mar, praia, floresta,
  montanha, cozinha, quarto, escritório, hospital, restaurante etc.
- Evitar pseudo-label por CLIP como fonte principal.
- Reduzir overfitting por:
    1. balanceamento por label principal;
    2. split determinístico por hash;
    3. limitação por classe/categoria;
    4. suporte a múltiplas fontes;
    5. CSV com metadados de origem.

Fontes suportadas:
- FiftyOne Places/Places365, quando disponível no ambiente.
- Diretório manual com imagens organizadas por categoria.

Exemplo de diretório manual:
data/external/places365/train/
  desert/
    img1.jpg
  street/
    img2.jpg

Ou estrutura aninhada:
data/external/places365/train/
  d/desert/sand/
    img1.jpg
  s/street/
    img2.jpg

O nome da categoria é inferido pelo caminho relativo da imagem.
"""


# ============================================================
# Labels novas recomendadas para SceneNet v2
# ============================================================



RECOMMENDED_SCENE_PLACE_LABELS = [
    "indoor",
    "outdoor",

    "room",
    "street",
    "road",
    "city",
    "desert",
    "beach",
    "ocean",
    "forest",
    "mountain",
    "park",
    "field",

    "school",
    "classroom",
    "kitchen",
    "bedroom",
    "living_room",
    "office",
    "restaurant",
    "store",
    "hospital",
    "sports_field",

    "day",
    "night",
    "dark_place",
    "bright_place",
]


# ============================================================
# Mapeamento de categorias Places/SUN/manual -> labels Spectra
# ============================================================

CATEGORY_RULES: Dict[str, List[str]] = {
    # Exterior genérico
    "street": ["street", "road", "city", "outdoor"],
    "alley": ["street", "city", "outdoor"],
    "road": ["road", "outdoor"],
    "highway": ["road", "outdoor"],
    "crosswalk": ["street", "road", "city", "outdoor"],
    "downtown": ["city", "street", "outdoor"],
    "skyscraper": ["city", "outdoor"],
    "city": ["city", "outdoor"],
    "urban": ["city", "street", "outdoor"],

    # Natureza / paisagem
    "desert": ["desert", "outdoor", "bright_place"],
    "sand": ["desert", "outdoor", "bright_place"],
    "dune": ["desert", "outdoor", "bright_place"],

    "beach": ["beach", "ocean", "outdoor", "bright_place"],
    "coast": ["beach", "ocean", "outdoor"],
    "coastline": ["beach", "ocean", "outdoor"],
    "shore": ["beach", "ocean", "outdoor"],
    "ocean": ["ocean", "outdoor"],
    "sea": ["ocean", "outdoor"],
    "water": ["ocean", "outdoor"],

    "forest": ["forest", "outdoor"],
    "woods": ["forest", "outdoor"],
    "rainforest": ["forest", "outdoor"],
    "jungle": ["forest", "outdoor"],

    "mountain": ["mountain", "outdoor"],
    "valley": ["mountain", "outdoor"],
    "cliff": ["mountain", "outdoor"],
    "canyon": ["mountain", "outdoor"],

    "park": ["park", "outdoor"],
    "garden": ["park", "outdoor"],
    "field": ["field", "outdoor"],
    "pasture": ["field", "outdoor"],
    "meadow": ["field", "outdoor"],
    "farm": ["field", "outdoor"],

    # Ambientes internos
    "room": ["room", "indoor"],
    "bedroom": ["bedroom", "room", "indoor"],
    "living_room": ["living_room", "room", "indoor"],
    "kitchen": ["kitchen", "room", "indoor"],
    "bathroom": ["room", "indoor"],
    "dining_room": ["restaurant", "room", "indoor"],

    "office": ["office_room", "room", "indoor"],
    "office_cubicles": ["office_cubicles", "room", "indoor"],
    "home_office": ["home_office", "room", "indoor"],
    "conference_room": ["conference_room", "room", "indoor"],
    "conference_room": ["office", "room", "indoor"],
    "cubicle": ["office", "room", "indoor"],

    "classroom": ["classroom", "school", "room", "indoor"],
    "school": ["school", "indoor"],
    "corridor": ["school", "hospital", "indoor"],

    "restaurant": ["restaurant_indoor", "room", "indoor"],
    "fastfood_restaurant": ["fastfood_restaurant", "room", "indoor"],
    "cafeteria": ["cafeteria", "room", "indoor"],
    "dining_room": ["dining_room", "room", "indoor"],
    "restaurant_patio": ["restaurant_patio", "outdoor"],
    "cafeteria": ["restaurant", "room", "indoor"],
    "bar": ["restaurant", "room", "indoor"],
    "coffee_shop": ["restaurant", "store", "indoor"],

    "store": ["store", "indoor"],
    "shop": ["store", "indoor"],
    "supermarket": ["store", "indoor"],
    "market": ["store", "indoor"],
    "mall": ["store", "indoor"],

    "hospital": ["hospital", "room", "indoor"],
    "hospital_room": ["hospital", "room", "indoor"],
    "operating_room": ["hospital", "room", "indoor"],

    "stadium": ["sports_field", "outdoor"],
    "football_field": ["sports_field", "field", "outdoor"],
    "soccer_field": ["sports_field", "field", "outdoor"],
    "baseball_field": ["sports_field", "field", "outdoor"],
    "athletic_field": ["sports_field", "field", "outdoor"],
    "sports_field": ["sports_field", "field", "outdoor"],

    # Iluminação aproximada por categoria
    "night": ["night", "dark_place"],
    "dark": ["dark_place"],
    "sunny": ["day", "bright_place"],
    "bright": ["bright_place"],
    "day": ["day", "bright_place"],
}

EXACT_CATEGORY_RULES = {
    # Office
    "/o/office": [
        "office_room",
        "room",
        "indoor",
    ],
    "/o/office_cubicles": [
        "office_cubicles",
        "room",
        "indoor",
    ],
    "/h/home_office": [
        "home_office",
        "room",
        "indoor",
    ],
    "/c/conference_room": [
        "conference_room",
        "room",
        "indoor",
    ],

    # Restaurant
    "/r/restaurant": [
        "restaurant_indoor",
        "room",
        "indoor",
    ],
    "/f/fastfood_restaurant": [
        "fastfood_restaurant",
        "room",
        "indoor",
    ],
    "/c/cafeteria": [
        "cafeteria",
        "room",
        "indoor",
    ],
    "/d/dining_room": [
        "dining_room",
        "room",
        "indoor",
    ],
    "/r/restaurant_patio": [
        "restaurant_patio",
        "outdoor",
    ],

    # Outros interiores
    "/b/bedroom": [
        "bedroom",
        "room",
        "indoor",
    ],
    "/l/living_room": [
        "living_room",
        "room",
        "indoor",
    ],
    "/k/kitchen": [
        "kitchen",
        "room",
        "indoor",
    ],

    # Escola
    "/c/classroom": [
        "classroom",
        "school",
        "room",
        "indoor",
    ],
    "/k/kindergarden_classroom": [
        "classroom",
        "school",
        "room",
        "indoor",
    ],
    "/k/kindergarten_classroom": [
        "classroom",
        "school",
        "room",
        "indoor",
    ],

    # Hospital
    "/h/hospital": [
        "hospital",
        "room",
        "indoor",
    ],
    "/h/hospital_room": [
        "hospital",
        "room",
        "indoor",
    ],
    "/o/operating_room": [
        "hospital",
        "room",
        "indoor",
    ],
}


CATEGORY_BLOCKLIST_BY_TARGET = {
    "office": [
        "office_building",
        "veterinarians_office",
    ],
    "restaurant": [
        "barndoor",
        "barn",
        "wet_bar",
        "bar",
        "restaurant_kitchen",
    ],
    "kitchen": [
        "restaurant_kitchen",
    ],
    "classroom": [],
    "hospital": [
        "corridor",
    ],
    "store": [
        "coffee_shop",
        "general_store/outdoor",
        "market/outdoor",
        "shopfront",
    ],
}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("\\", "/")
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = value.replace(".", "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_/")

def normalize_category_path(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("\\", "/")
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = value.replace(".", "_")

    while "//" in value:
        value = value.replace("//", "/")

    while "__" in value:
        value = value.replace("__", "_")

    value = value.strip()

    if not value.startswith("/"):
        value = "/" + value

    return value.rstrip("/")

def stable_hash(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def deterministic_split(
    key: str,
    train_ratio: float,
    validation_ratio: float,
) -> str:
    bucket = stable_hash(key) % 10000
    train_limit = int(train_ratio * 10000)
    validation_limit = int((train_ratio + validation_ratio) * 10000)

    if bucket < train_limit:
        return "train"

    if bucket < validation_limit:
        return "validation"

    return "test"


def get_active_scene_labels() -> List[str]:
    return list(SPECTRA_SCENE_LABELS)


def warn_missing_recommended_labels(active_labels: Sequence[str]) -> None:
    active_set = set(active_labels)
    missing = [
        label
        for label in RECOMMENDED_SCENE_PLACE_LABELS
        if label not in active_set
    ]

    if missing:
        print("\nAVISO: estas labels recomendadas não existem em SPECTRA_SCENE_LABELS:")
        print(missing)
        print(
            "Se quiser treinar essas classes, adicione-as em ai/spectra/labels/label_sets.py."
        )


def category_to_spectra_labels(
    category_name: str,
    active_labels: Sequence[str],
) -> List[str]:
    normalized_path = normalize_category_path(category_name)
    normalized_category = normalize_text(category_name)

    active_label_set = set(active_labels)

    # ============================================================
    # 1. Regra exata: se a categoria do Places é conhecida,
    #    usa somente o mapeamento manual e não cai no substring.
    # ============================================================
    if normalized_path in EXACT_CATEGORY_RULES:
        exact_labels = []

        for label in EXACT_CATEGORY_RULES[normalized_path]:
            if label in active_label_set:
                exact_labels.append(label)

        return sorted(set(exact_labels))

    labels = set()

    category_parts = set(normalized_category.split("/"))

    tokens = set()
    for part in normalized_category.replace("/", "_").split("_"):
        if part:
            tokens.add(part)

    # ============================================================
    # 2. Regra genérica: fallback para categorias não listadas.
    # ============================================================
    for pattern, mapped_labels in CATEGORY_RULES.items():
        normalized_pattern = normalize_text(pattern)

        pattern_matched = (
            normalized_pattern in category_parts
            or normalized_pattern in tokens
        )

        # Evita substring solta.
        # Antes: "garden" dentro de "kindergarden_classroom" ativava park.
        # Agora só aceita partes/tokens.
        if not pattern_matched:
            continue

        for label in mapped_labels:
            if label in active_label_set:
                labels.add(label)

    # ============================================================
    # 3. Blocklist por target:
    #    remove labels específicas quando a categoria é conhecida
    #    como ruim/ambígua para aquela label.
    # ============================================================
    for target_label, blocked_patterns in CATEGORY_BLOCKLIST_BY_TARGET.items():
        if target_label not in labels:
            continue

        for blocked_pattern in blocked_patterns:
            normalized_blocked = normalize_category_path(blocked_pattern)
            normalized_blocked_text = normalize_text(blocked_pattern)

            is_blocked = (
                normalized_blocked == normalized_path
                or normalized_blocked_text in category_parts
                or normalized_blocked_text in tokens
                or normalized_blocked_text == normalized_category
            )

            if is_blocked:
                labels.discard(target_label)

    # ============================================================
    # 4. Se só sobrou room/indoor/outdoor sem classe específica,
    #    ainda pode ser útil, mas não para reforço de target.
    # ============================================================

    if "indoor" in labels and "outdoor" in labels:
        indoor_words = [
            "room",
            "kitchen",
            "bedroom",
            "office",
            "classroom",
            "hospital",
            "restaurant",
            "store",
            "indoor",
        ]
        outdoor_words = [
            "street",
            "road",
            "city",
            "desert",
            "beach",
            "ocean",
            "forest",
            "mountain",
            "park",
            "field",
            "outdoor",
        ]

        has_indoor = any(word in normalized_category for word in indoor_words)
        has_outdoor = any(word in normalized_category for word in outdoor_words)

        if has_indoor and not has_outdoor:
            labels.discard("outdoor")
        elif has_outdoor and not has_indoor:
            labels.discard("indoor")
        else:
            labels.discard("indoor")
            labels.discard("outdoor")

    return sorted(labels)


def infer_primary_label(labels: Sequence[str]) -> Optional[str]:
    priority = [
        "desert",
        "beach",
        "ocean",
        "forest",
        "mountain",
        "street",
        "road",
        "city",
        "park",
        "field",

        "kitchen",
        "bedroom",
        "living_room",

        "office_room",
        "office_cubicles",
        "home_office",
        "conference_room",

        "restaurant_indoor",
        "fastfood_restaurant",
        "cafeteria",
        "dining_room",
        "restaurant_patio",

        "classroom",
        "school",
        "store",
        "hospital",
        "sports_field",
        "room",
        "indoor",
        "outdoor",
    ]

    label_set = set(labels)

    for label in priority:
        if label in label_set:
            return label

    if labels:
        return labels[0]

    return None


def build_label_row(
    image_path: Path,
    labels: Sequence[str],
    active_labels: Sequence[str],
    source_dataset: str,
    source_category: str,
    source_split: str,
    generated_split: str,
) -> Dict[str, object]:
    label_set = set(labels)

    row = {
        "frame_path": str(image_path),
        "source_dataset": source_dataset,
        "source_category": source_category,
        "source_split": source_split,
        "generated_split": generated_split,
        "primary_label": infer_primary_label(labels) or "",
    }

    for label in active_labels:
        row[label] = 1 if label in label_set else 0

    return row


def iter_images_from_directory(input_dir: Path) -> Iterable[Tuple[Path, str]]:
    input_dir = input_dir.resolve()

    for image_path in sorted(input_dir.rglob("*")):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative = image_path.relative_to(input_dir)

        if len(relative.parts) <= 1:
            category_name = image_path.parent.name
        else:
            category_name = "/".join(relative.parts[:-1])

        yield image_path, category_name


def load_fiftyone_samples(
    dataset_name: str,
    split: str,
    max_samples: Optional[int],
    fiftyone_name: str,
):
    try:
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise ImportError(
            "FiftyOne não está instalado. Instale com: pip install fiftyone"
        ) from exc

    kwargs = {
        "split": split,
        "dataset_name": fiftyone_name,
        "shuffle": True,
    }

    if max_samples is not None:
        kwargs["max_samples"] = max_samples

    last_error = None

    for candidate_name in [dataset_name, "places365", "places"]:
        try:
            dataset = foz.load_zoo_dataset(candidate_name, **kwargs)
            return dataset, candidate_name
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Não consegui carregar Places via FiftyOne. Último erro: {}".format(last_error)
    )


def extract_fiftyone_label(sample) -> Optional[str]:
    candidate_fields = [
        "ground_truth",
        "label",
        "classification",
        "scene",
    ]

    for field in candidate_fields:
        try:
            if not sample.has_field(field):
                continue

            value = sample[field]

            if hasattr(value, "label"):
                return str(value.label)

            if isinstance(value, str):
                return value

        except Exception:
            continue

    # Fallback: algumas versões guardam em tags.
    try:
        if sample.tags:
            return str(sample.tags[0])
    except Exception:
        pass

    return None


def collect_rows_from_fiftyone(
    dataset_name: str,
    split: str,
    max_samples: Optional[int],
    active_labels: Sequence[str],
    train_ratio: float,
    validation_ratio: float,
    fiftyone_name: str,
) -> List[Dict[str, object]]:
    dataset, resolved_name = load_fiftyone_samples(
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples,
        fiftyone_name=fiftyone_name,
    )

    rows = []
    skipped_without_label = 0
    skipped_unmapped = 0

    for sample in dataset:
        category_name = extract_fiftyone_label(sample)

        if not category_name:
            skipped_without_label += 1
            continue

        labels = category_to_spectra_labels(category_name, active_labels)

        if not labels:
            skipped_unmapped += 1
            continue

        image_path = Path(sample.filepath)

        generated_split = deterministic_split(
            key=str(image_path),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )

        rows.append(
            build_label_row(
                image_path=image_path,
                labels=labels,
                active_labels=active_labels,
                source_dataset=resolved_name,
                source_category=category_name,
                source_split=split,
                generated_split=generated_split,
            )
        )

    print("\nFiftyOne carregado:", resolved_name)
    print("Amostras aceitas:", len(rows))
    print("Sem label:", skipped_without_label)
    print("Categorias sem mapeamento:", skipped_unmapped)

    return rows


def collect_rows_from_directory(
    input_dir: Path,
    source_dataset: str,
    source_split: str,
    max_samples: Optional[int],
    active_labels: Sequence[str],
    train_ratio: float,
    validation_ratio: float,
) -> List[Dict[str, object]]:
    rows = []
    skipped_unmapped = 0

    for image_path, category_name in iter_images_from_directory(input_dir):
        labels = category_to_spectra_labels(category_name, active_labels)

        if not labels:
            skipped_unmapped += 1
            continue

        generated_split = deterministic_split(
            key=str(image_path),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )

        rows.append(
            build_label_row(
                image_path=image_path,
                labels=labels,
                active_labels=active_labels,
                source_dataset=source_dataset,
                source_category=category_name,
                source_split=source_split,
                generated_split=generated_split,
            )
        )

        if max_samples is not None and len(rows) >= max_samples:
            break

    print("\nDiretório carregado:", input_dir)
    print("Amostras aceitas:", len(rows))
    print("Categorias sem mapeamento:", skipped_unmapped)

    return rows


def balance_rows(
    rows: List[Dict[str, object]],
    max_per_primary_label: Optional[int],
    seed: int,
) -> List[Dict[str, object]]:
    if max_per_primary_label is None:
        return rows

    random.seed(seed)

    buckets = defaultdict(list)

    for row in rows:
        primary_label = row.get("primary_label") or "unknown"
        buckets[primary_label].append(row)

    balanced = []

    for primary_label, bucket_rows in sorted(buckets.items()):
        random.shuffle(bucket_rows)
        selected = bucket_rows[:max_per_primary_label]
        balanced.extend(selected)

    random.shuffle(balanced)

    return balanced


def copy_images_if_requested(
    rows: List[Dict[str, object]],
    output_images_dir: Optional[Path],
    copy_images: bool,
) -> List[Dict[str, object]]:
    if not copy_images:
        return rows

    if output_images_dir is None:
        raise ValueError("--copy-images exige --output-images-dir.")

    output_images_dir.mkdir(parents=True, exist_ok=True)

    copied_rows = []

    for index, row in enumerate(rows):
        source_path = Path(str(row["frame_path"]))

        if not source_path.exists():
            continue

        source_category = normalize_text(str(row.get("source_category", "unknown")))
        extension = source_path.suffix.lower()
        filename = "{}_{}{}".format(index, stable_hash(str(source_path)), extension)

        category_dir = output_images_dir / source_category.replace("/", "_")
        category_dir.mkdir(parents=True, exist_ok=True)

        output_path = category_dir / filename

        if not output_path.exists():
            shutil.copy2(source_path, output_path)

        new_row = dict(row)
        new_row["frame_path"] = str(output_path)
        new_row["source_original_path"] = str(source_path)
        copied_rows.append(new_row)

    return copied_rows


def save_report(
    rows: List[Dict[str, object]],
    output_report: Path,
    active_labels: Sequence[str],
) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)

    label_counts = {
        label: 0
        for label in active_labels
    }

    primary_counts = defaultdict(int)
    category_counts = defaultdict(int)
    split_counts = defaultdict(int)

    for row in rows:
        primary_counts[str(row.get("primary_label", ""))] += 1
        category_counts[str(row.get("source_category", ""))] += 1
        split_counts[str(row.get("generated_split", ""))] += 1

        for label in active_labels:
            label_counts[label] += int(row.get(label, 0))

    report = {
        "total_rows": len(rows),
        "active_labels": list(active_labels),
        "recommended_missing_from_label_set": [
            label
            for label in RECOMMENDED_SCENE_PLACE_LABELS
            if label not in active_labels
        ],
        "label_counts": dict(sorted(label_counts.items())),
        "primary_label_counts": dict(sorted(primary_counts.items())),
        "source_category_counts": dict(sorted(category_counts.items())),
        "generated_split_counts": dict(sorted(split_counts.items())),
        "category_rules": CATEGORY_RULES,
    }

    with open(output_report, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


def print_summary(rows: List[Dict[str, object]], active_labels: Sequence[str]) -> None:
    print("\nResumo do dataset SceneNet v2")
    print("Total:", len(rows))

    if not rows:
        return

    dataframe = pd.DataFrame(rows)

    metadata_cols = [
        "frame_path",
        "source_dataset",
        "source_category",
        "source_split",
        "generated_split",
        "primary_label",
        "source_original_path",
    ]

    label_cols = [
        label
        for label in active_labels
        if label in dataframe.columns
    ]

    counts = dataframe[label_cols].sum().sort_values(ascending=False)

    print("\nLabels mais frequentes:")
    print(counts.head(50))

    print("\nLabels zeradas:")
    print(list(counts[counts == 0].index))

    print("\nMédia de labels positivas por imagem:")
    print(dataframe[label_cols].sum(axis=1).mean())

    print("\nDistribuição dos splits gerados:")
    print(dataframe["generated_split"].value_counts())

    print("\nPrimary labels:")
    print(dataframe["primary_label"].value_counts().head(50))

def parse_target_labels(value: Optional[str]) -> List[str]:
    if not value:
        return []

    labels = []

    for item in value.split(","):
        label = item.strip()

        if label:
            labels.append(label)

    return labels


def load_existing_rows_if_available(
    output_csv: str,
    active_labels: Sequence[str],
) -> List[Dict[str, object]]:
    output_path = Path(output_csv)

    if not output_path.exists():
        return []

    dataframe = pd.read_csv(output_path)

    print("\nCSV existente encontrado:")
    print(output_path)
    print("Linhas existentes:", len(dataframe))

    rows = dataframe.to_dict(orient="records")

    normalized_rows = []

    for row in rows:
        normalized = dict(row)

        for label in active_labels:
            if label not in normalized:
                normalized[label] = 0

            normalized[label] = int(normalized[label])

        normalized_rows.append(normalized)

    return normalized_rows


def dedupe_rows_by_frame_path(
    rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    seen = set()
    deduped = []

    for row in rows:
        frame_path = str(row.get("frame_path", ""))

        if not frame_path:
            continue

        if frame_path in seen:
            continue

        seen.add(frame_path)
        deduped.append(row)

    return deduped


def count_labels(
    rows: List[Dict[str, object]],
    labels: Sequence[str],
) -> Dict[str, int]:
    counts = {
        label: 0
        for label in labels
    }

    for row in rows:
        for label in labels:
            counts[label] += int(row.get(label, 0))

    return counts


def get_missing_target_counts(
    rows: List[Dict[str, object]],
    target_labels: Sequence[str],
    min_per_target_label: int,
) -> Dict[str, int]:
    counts = count_labels(rows, target_labels)

    missing = {}

    for label in target_labels:
        current = counts.get(label, 0)
        remaining = max(0, min_per_target_label - current)

        if remaining > 0:
            missing[label] = remaining

    return missing


def row_matches_target_labels(
    row: Dict[str, object],
    target_labels: Sequence[str],
) -> bool:
    if not target_labels:
        return True

    for label in target_labels:
        if int(row.get(label, 0)) == 1:
            return True

    return False


def select_rows_to_improve_targets(
    candidate_rows: List[Dict[str, object]],
    existing_rows: List[Dict[str, object]],
    target_labels: Sequence[str],
    min_per_target_label: int,
    max_new_rows: Optional[int],
    seed: int,
) -> List[Dict[str, object]]:
    if not target_labels:
        if max_new_rows is None:
            return candidate_rows

        return candidate_rows[:max_new_rows]

    random.seed(seed)

    existing_paths = {
        str(row.get("frame_path", ""))
        for row in existing_rows
    }

    current_counts = count_labels(existing_rows, target_labels)

    shuffled_candidates = list(candidate_rows)
    random.shuffle(shuffled_candidates)

    selected = []

    for row in shuffled_candidates:
        if max_new_rows is not None and len(selected) >= max_new_rows:
            break

        frame_path = str(row.get("frame_path", ""))

        if not frame_path or frame_path in existing_paths:
            continue

        useful = False

        for label in target_labels:
            if int(row.get(label, 0)) != 1:
                continue

            if current_counts.get(label, 0) < min_per_target_label:
                useful = True
                break

        if not useful:
            continue

        selected.append(row)
        existing_paths.add(frame_path)

        for label in target_labels:
            current_counts[label] = current_counts.get(label, 0) + int(row.get(label, 0))

        all_targets_reached = all(
            current_counts.get(label, 0) >= min_per_target_label
            for label in target_labels
        )

        if all_targets_reached:
            break

    print("\nReforço por target_labels:")
    print("Targets:", list(target_labels))
    print("Mínimo desejado por target:", min_per_target_label)
    print("Novas linhas selecionadas:", len(selected))

    print("\nContagem após seleção:")
    for label in target_labels:
        print(f"{label}: {current_counts.get(label, 0)}")

    still_missing = {
        label: max(0, min_per_target_label - current_counts.get(label, 0))
        for label in target_labels
    }

    still_missing = {
        label: value
        for label, value in still_missing.items()
        if value > 0
    }

    if still_missing:
        print("\nAinda faltam exemplos:")
        print(still_missing)

    return selected


def build_spectra_scene_dataset_places(
    source: str,
    output_csv: str,
    output_report: str,
    input_dir: Optional[str] = None,
    source_dataset: str = "places365_manual",
    source_split: str = "train",
    fiftyone_dataset_name: str = "places365",
    fiftyone_name: str = "spectra_places365_scene",
    max_samples: Optional[int] = None,
    max_per_primary_label: Optional[int] = 500,
    output_images_dir: Optional[str] = None,
    copy_images: bool = False,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    seed: int = 42,
    target_labels: Optional[Sequence[str]] = None,
    min_per_target_label: int = 250,
    max_new_rows: Optional[int] = None,
    append_existing: bool = True,
) -> Path:
    active_labels = get_active_scene_labels()
    warn_missing_recommended_labels(active_labels)

    target_labels = list(target_labels or [])

    invalid_targets = [
        label
        for label in target_labels
        if label not in active_labels
    ]

    if invalid_targets:
        raise ValueError(
            "Estas target_labels não existem em SPECTRA_SCENE_LABELS: {}".format(
                invalid_targets
            )
        )

    existing_rows = []

    if append_existing:
        existing_rows = load_existing_rows_if_available(
            output_csv=output_csv,
            active_labels=active_labels,
        )

    if target_labels and existing_rows:
        missing = get_missing_target_counts(
            rows=existing_rows,
            target_labels=target_labels,
            min_per_target_label=min_per_target_label,
        )

        print("\nTargets abaixo do mínimo antes da busca:")
        print(missing if missing else "Nenhum. Todos já atingiram o mínimo.")

        if not missing:
            print("\nNenhum reforço necessário. Mantendo CSV existente.")
            rows = dedupe_rows_by_frame_path(existing_rows)

            output_csv_path = Path(output_csv)
            dataframe = pd.DataFrame(rows)
            dataframe.to_csv(output_csv_path, index=False)

            save_report(
                rows=rows,
                output_report=Path(output_report),
                active_labels=active_labels,
            )

            print_summary(rows, active_labels)

            return output_csv_path

    if source == "fiftyone":
        candidate_rows = collect_rows_from_fiftyone(
            dataset_name=fiftyone_dataset_name,
            split=source_split,
            max_samples=max_samples,
            active_labels=active_labels,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            fiftyone_name=fiftyone_name,
        )

    elif source == "directory":
        if input_dir is None:
            raise ValueError("--input-dir é obrigatório quando --source directory.")

        candidate_rows = collect_rows_from_directory(
            input_dir=Path(input_dir),
            source_dataset=source_dataset,
            source_split=source_split,
            max_samples=max_samples,
            active_labels=active_labels,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )

    else:
        raise ValueError("--source deve ser 'fiftyone' ou 'directory'.")

    if target_labels:
        new_rows = select_rows_to_improve_targets(
            candidate_rows=candidate_rows,
            existing_rows=existing_rows,
            target_labels=target_labels,
            min_per_target_label=min_per_target_label,
            max_new_rows=max_new_rows,
            seed=seed,
        )
    else:
        new_rows = candidate_rows

        if max_per_primary_label is not None:
            new_rows = balance_rows(
                rows=new_rows,
                max_per_primary_label=max_per_primary_label,
                seed=seed,
            )

    new_rows = copy_images_if_requested(
        rows=new_rows,
        output_images_dir=Path(output_images_dir) if output_images_dir else None,
        copy_images=copy_images,
    )

    rows = existing_rows + new_rows
    rows = dedupe_rows_by_frame_path(rows)

    if not target_labels and max_per_primary_label is not None:
        rows = balance_rows(
            rows=rows,
            max_per_primary_label=max_per_primary_label,
            seed=seed,
        )

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_csv_path, index=False)

    save_report(
        rows=rows,
        output_report=Path(output_report),
        active_labels=active_labels,
    )

    print_summary(rows, active_labels)

    print("\nCSV salvo em:", output_csv_path)
    print("Relatório salvo em:", output_report)

    if target_labels:
        final_counts = count_labels(rows, target_labels)

        print("\nContagem final das target_labels:")
        for label in target_labels:
            print(f"{label}: {final_counts.get(label, 0)}")

    return output_csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Cria dataset otimizado de cenas/lugares para SpectraSceneNet v2."
    )

    parser.add_argument(
        "--source",
        choices=["fiftyone", "directory"],
        required=True,
        help="Fonte dos dados: fiftyone ou directory.",
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help="Diretório manual com imagens organizadas por categoria.",
    )

    parser.add_argument(
        "--source-dataset",
        default="places365_manual",
        help="Nome da fonte usado nos metadados quando --source directory.",
    )

    parser.add_argument(
        "--source-split",
        default="train",
        help="Split original da fonte. Ex.: train, validation, test.",
    )

    parser.add_argument(
        "--fiftyone-dataset-name",
        default="places365",
        help="Nome tentado no FiftyOne Zoo. O script também tenta places365 e places.",
    )

    parser.add_argument(
        "--fiftyone-name",
        default="spectra_places365_scene",
        help="Nome local do dataset no FiftyOne.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limite total de amostras antes do balanceamento.",
    )

    parser.add_argument(
        "--max-per-primary-label",
        type=int,
        default=500,
        help="Máximo por primary_label. Use 0 para desativar.",
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/spectra_scene_places_labels.csv",
        help="CSV final no formato da Spectra.",
    )

    parser.add_argument(
        "--output-report",
        default="data/datasets/spectra_scene_places_report.json",
        help="Relatório JSON de distribuição e mapeamento.",
    )

    parser.add_argument(
        "--output-images-dir",
        default=None,
        help="Pasta para copiar imagens usadas, se --copy-images estiver ativo.",
    )

    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copia imagens para uma pasta local controlada.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Proporção de treino no split gerado por hash.",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
        help="Proporção de validação no split gerado por hash.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para balanceamento.",
    )

    parser.add_argument(
        "--target-labels",
        default=None,
        help=(
            "Labels específicas para reforçar, separadas por vírgula. "
            "Ex.: living_room,bedroom,office,restaurant,kitchen"
        ),
    )

    parser.add_argument(
        "--min-per-target-label",
        type=int,
        default=250,
        help="Mínimo desejado de exemplos para cada target label.",
    )

    parser.add_argument(
        "--max-new-rows",
        type=int,
        default=None,
        help="Máximo de novas linhas adicionadas ao CSV existente.",
    )

    parser.add_argument(
        "--no-append-existing",
        action="store_true",
        help="Não carrega o CSV existente; recria do zero.",
    )

    args = parser.parse_args()

    max_per_primary_label = args.max_per_primary_label

    if max_per_primary_label == 0:
        max_per_primary_label = None

    build_spectra_scene_dataset_places(
    source=args.source,
    output_csv=args.output_csv,
    output_report=args.output_report,
    input_dir=args.input_dir,
    source_dataset=args.source_dataset,
    source_split=args.source_split,
    fiftyone_dataset_name=args.fiftyone_dataset_name,
    fiftyone_name=args.fiftyone_name,
    max_samples=args.max_samples,
    max_per_primary_label=max_per_primary_label,
    output_images_dir=args.output_images_dir,
    copy_images=args.copy_images,
    train_ratio=args.train_ratio,
    validation_ratio=args.validation_ratio,
    seed=args.seed,
    target_labels=parse_target_labels(args.target_labels),
    min_per_target_label=args.min_per_target_label,
    max_new_rows=args.max_new_rows,
    append_existing=not args.no_append_existing,
    )


if __name__ == "__main__":
    main()