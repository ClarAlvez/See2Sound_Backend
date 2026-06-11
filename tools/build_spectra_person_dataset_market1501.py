import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy.io import loadmat

from ai.spectra.labels.label_sets import SPECTRA_PERSON_LABELS


"""
Builder para SpectraPersonNet usando Market-1501 + market_attribute.mat.

Entrada esperada:
data/external/market1501/
  Market-1501-v15.09.15/
    bounding_box_train/
    bounding_box_test/
    query/
  market_attribute.mat

Saída:
data/datasets/spectra_person_labels.csv

Observação:
O Market-1501 Attribute normalmente possui atributos por person_id,
não por imagem individual.
"""


MARKET_TO_SPECTRA_CANDIDATES: Dict[str, List[str]] = {
    "person": ["person"],

    "man": ["gender_male", "male", "man"],
    "woman": ["gender_female", "female", "woman"],

    "backpack": ["backpack", "back_pack"],
    "bag": ["bag", "handbag", "shoulderbag", "shoulder_bag"],
    "hat": ["hat"],
    "cap": ["cap"],
    "glasses": ["glasses"],

    "black_clothes": [
        "upblack",
        "downblack",
        "upper_black",
        "lower_black",
        "black_upper",
        "black_lower",
        "black_clothes",
    ],
    "white_clothes": [
        "upwhite",
        "downwhite",
        "upper_white",
        "lower_white",
        "white_upper",
        "white_lower",
        "white_clothes",
    ],
    "red_clothes": [
        "upred",
        "downred",
        "upper_red",
        "lower_red",
        "red_upper",
        "red_lower",
        "red_clothes",
    ],
    "blue_clothes": [
        "upblue",
        "downblue",
        "upper_blue",
        "lower_blue",
        "blue_upper",
        "blue_lower",
        "blue_clothes",
    ],
    "green_clothes": [
        "upgreen",
        "downgreen",
        "upper_green",
        "lower_green",
        "green_upper",
        "green_lower",
        "green_clothes",
    ],
    "yellow_clothes": [
        "upyellow",
        "downyellow",
        "upper_yellow",
        "lower_yellow",
        "yellow_upper",
        "yellow_lower",
        "yellow_clothes",
    ],

    "dress": ["dress", "skirt"],
    "shirt": ["shirt", "tshirt", "t_shirt"],
    "jacket": ["jacket", "coat"],

    "short_hair": ["hair_short", "short_hair", "short"],
    "long_hair": ["hair_long", "long_hair", "long"],

    # Normalmente Market Attribute não possui cor de cabelo confiável.
    "blonde_hair": ["blonde_hair", "hair_blonde"],
    "brown_hair": ["brown_hair", "hair_brown"],
    "black_hair": ["black_hair", "hair_black"],
    "red_hair": ["red_hair", "hair_red"],
    "gray_hair": ["gray_hair", "hair_gray", "grey_hair"],

    # Caso você tenha renomeado no label_sets.py:
    "light_skin_tone": ["light_skin_tone", "skin_tone_light", "light_skin"],
    "medium_skin_tone": ["medium_skin_tone", "skin_tone_medium", "medium_skin"],
    "dark_skin_tone": ["dark_skin_tone", "skin_tone_dark", "dark_skin"],

    # Caso ainda esteja com nomes antigos:
    "light_skin": ["light_skin", "light_skin_tone", "skin_tone_light"],
    "medium_skin": ["medium_skin", "medium_skin_tone", "skin_tone_medium"],
    "dark_skin": ["dark_skin", "dark_skin_tone", "skin_tone_dark"],

    # Market não costuma ter essas labels diretamente.
    "face_visible": ["face_visible"],
    "hand_visible": ["hand_visible"],
    "child": ["child"],
    "curly_hair": ["curly_hair", "hair_curly"],
    "straight_hair": ["straight_hair", "hair_straight"],
}


MUTUALLY_EXCLUSIVE_GROUPS = [
    ["man", "woman", "child"],
    ["short_hair", "long_hair"],
    ["light_skin", "medium_skin", "dark_skin"],
    ["light_skin_tone", "medium_skin_tone", "dark_skin_tone"],
]


def normalize_name(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = value.replace("/", "_")
    value = value.replace(".", "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")


def extract_person_id_from_filename(image_path: Path) -> Optional[int]:
    """
    Exemplo:
    0002_c1s1_000451_03.jpg -> 2
    -1_c1s1_... -> inválido
    """
    first_part = image_path.name.split("_")[0]

    try:
        person_id = int(first_part)
    except ValueError:
        return None

    if person_id < 0:
        return None

    return person_id


def unwrap_mat_struct(value):
    """
    Ajuda a acessar estruturas MATLAB carregadas pelo scipy.
    """
    while hasattr(value, "shape") and value.shape == (1, 1):
        value = value[0, 0]

    return value


def mat_field_names(mat_struct) -> List[str]:
    if hasattr(mat_struct, "dtype") and mat_struct.dtype.names:
        return list(mat_struct.dtype.names)

    return []


def get_field(mat_struct, field_name):
    value = mat_struct[field_name]
    return unwrap_mat_struct(value)


def extract_market_split_attributes(market_attribute, split_name: str):
    """
    Tenta extrair o split 'train' ou 'test' de várias estruturas possíveis.
    """
    market_attribute = unwrap_mat_struct(market_attribute)

    fields = mat_field_names(market_attribute)

    if split_name in fields:
        return unwrap_mat_struct(get_field(market_attribute, split_name))

    # Alguns arquivos podem ter nomes alternativos.
    alternatives = {
        "train": ["train", "training"],
        "test": ["test", "testing"],
    }

    for alternative in alternatives.get(split_name, []):
        if alternative in fields:
            return unwrap_mat_struct(get_field(market_attribute, alternative))

    raise ValueError(
        "Não encontrei o split '{}' dentro de market_attribute. Campos disponíveis: {}".format(
            split_name,
            fields,
        )
    )


def extract_attribute_table(split_struct) -> pd.DataFrame:
    split_struct = unwrap_mat_struct(split_struct)
    fields = mat_field_names(split_struct)

    if not fields:
        raise ValueError("Estrutura do split não possui campos MATLAB nomeados.")

    raw_data = {}

    for field in fields:
        value = get_field(split_struct, field)

        try:
            flattened = value.squeeze()
        except AttributeError:
            flattened = value

        normalized_field = normalize_name(field)

        if normalized_field == "image_index":
            parsed_values = []

            for item in flattened:
                try:
                    if hasattr(item, "__len__") and not isinstance(item, str):
                        parsed_values.append(str(item[0]).strip())
                    else:
                        parsed_values.append(str(item).strip())
                except Exception:
                    parsed_values.append(str(item).strip())

            raw_data[normalized_field] = parsed_values

        else:
            raw_data[normalized_field] = [
                int(item)
                for item in flattened
            ]

    dataframe = pd.DataFrame(raw_data)

    if "image_index" not in dataframe.columns:
        raise ValueError("O split não possui campo image_index.")

    dataframe["person_id"] = dataframe["image_index"].astype(str).str.zfill(4).astype(int)

    return dataframe


def load_market_attribute_dataframe(
    market_attribute_path: Path,
    split_name: str,
) -> pd.DataFrame:
    mat_data = loadmat(str(market_attribute_path))

    if "market_attribute" in mat_data:
        market_attribute = mat_data["market_attribute"]
    else:
        keys = [key for key in mat_data.keys() if not key.startswith("__")]
        if len(keys) == 1:
            market_attribute = mat_data[keys[0]]
        else:
            raise ValueError(
                "Não encontrei 'market_attribute'. Chaves disponíveis: {}".format(keys)
            )

    split_struct = extract_market_split_attributes(
        market_attribute=market_attribute,
        split_name=split_name,
    )

    return extract_attribute_table(split_struct)


def is_market_positive(value) -> bool:
    """
    No Market-1501 Attribute:
    1 = não / ausência
    2 = sim / presença
    """
    try:
        return int(value) == 2
    except Exception:
        return False


def build_labels_from_attributes(attribute_row: pd.Series) -> Dict[str, int]:
    labels = {
        label: 0
        for label in SPECTRA_PERSON_LABELS
    }

    if "person" in labels:
        labels["person"] = 1

    # ======================================================
    # GENDER
    # Market:
    # gender = 1 -> man
    # gender = 2 -> woman
    # ======================================================
    if "gender" in attribute_row.index:
        try:
            gender = int(attribute_row["gender"])

            if gender == 1 and "man" in labels:
                labels["man"] = 1

            if gender == 2 and "woman" in labels:
                labels["woman"] = 1

        except Exception:
            pass

    # ======================================================
    # HAIR
    # Market:
    # hair = 1 -> short_hair
    # hair = 2 -> long_hair
    # ======================================================
    if "hair" in attribute_row.index:
        try:
            hair = int(attribute_row["hair"])

            if hair == 1 and "short_hair" in labels:
                labels["short_hair"] = 1

            if hair == 2 and "long_hair" in labels:
                labels["long_hair"] = 1

        except Exception:
            pass

    # ======================================================
    # BINARY ATTRIBUTES
    # 2 = positivo
    # 1 = negativo
    # ======================================================
    binary_mapping = {
        "backpack": "backpack",
        "bag": "bag",
        "handbag": "bag",

        "hat": "hat",

        "upblack": "black_clothes",
        "downblack": "black_clothes",

        "upwhite": "white_clothes",
        "downwhite": "white_clothes",

        "upred": "red_clothes",

        "upblue": "blue_clothes",
        "downblue": "blue_clothes",

        "upgreen": "green_clothes",
        "downgreen": "green_clothes",

        "upyellow": "yellow_clothes",
        "downyellow": "yellow_clothes",
    }

    for market_label, spectra_label in binary_mapping.items():
        if spectra_label not in labels:
            continue

        if market_label not in attribute_row.index:
            continue

        if is_market_positive(attribute_row[market_label]):
            labels[spectra_label] = 1

    apply_consistency_rules(labels)

    return labels


def apply_consistency_rules(labels: Dict[str, int]) -> None:
    def keep_only_first_active(group: List[str]) -> None:
        existing = [label for label in group if label in labels]
        active = [label for label in existing if labels.get(label, 0) == 1]

        if len(active) <= 1:
            return

        best = active[0]

        for label in active:
            labels[label] = 1 if label == best else 0

    for group in MUTUALLY_EXCLUSIVE_GROUPS:
        keep_only_first_active(group)

    if "person" in labels:
        labels["person"] = 1


def build_output_row(
    image_path: Path,
    labels: Dict[str, int],
    source_split: str,
    person_id: int,
) -> Dict[str, object]:
    row = {
        "frame_path": str(image_path),
        "source_dataset": "market1501_attribute",
        "source_split": source_split,
        "source_person_id": person_id,
    }

    for label in SPECTRA_PERSON_LABELS:
        row[label] = int(labels.get(label, 0))

    return row


def save_mapping_report(
    output_report: Path,
    attribute_dataframe: pd.DataFrame,
) -> None:
    import json

    columns = list(attribute_dataframe.columns)

    report = {
        "available_attribute_columns": columns,
        "spectra_person_labels": SPECTRA_PERSON_LABELS,
        "mapped_labels": {},
        "unmapped_labels": [],
    }

    available = set(columns)

    for spectra_label in SPECTRA_PERSON_LABELS:
        candidates = MARKET_TO_SPECTRA_CANDIDATES.get(spectra_label, [])
        matched = []

        for candidate in candidates:
            normalized_candidate = normalize_name(candidate)
            if normalized_candidate in available:
                matched.append(normalized_candidate)

        if matched:
            report["mapped_labels"][spectra_label] = matched
        else:
            report["unmapped_labels"].append(spectra_label)

    output_report.parent.mkdir(parents=True, exist_ok=True)

    with open(output_report, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


def build_market1501_person_dataset(
    market_root_dir: str,
    market_attribute_path: str,
    output_csv: str,
    split: str = "train",
    max_images: Optional[int] = None,
    output_report: str = "data/datasets/spectra_person_market1501_mapping_report.json",
) -> Path:
    market_root_dir = Path(market_root_dir)
    market_attribute_path = Path(market_attribute_path)
    output_csv = Path(output_csv)
    output_report = Path(output_report)

    if split == "train":
        image_dir = market_root_dir / "bounding_box_train"
        attribute_split = "train"
    elif split == "test":
        image_dir = market_root_dir / "bounding_box_test"
        attribute_split = "test"
    elif split == "query":
        image_dir = market_root_dir / "query"
        attribute_split = "test"
    else:
        raise ValueError("Split inválido: {}. Use train, test ou query.".format(split))

    if not image_dir.exists():
        raise FileNotFoundError("Pasta de imagens não encontrada: {}".format(image_dir))

    if not market_attribute_path.exists():
        raise FileNotFoundError("Arquivo market_attribute não encontrado: {}".format(market_attribute_path))

    attribute_dataframe = load_market_attribute_dataframe(
        market_attribute_path=market_attribute_path,
        split_name=attribute_split,
    )

    save_mapping_report(
        output_report=output_report,
        attribute_dataframe=attribute_dataframe,
    )

    attributes_by_person_id = {
        int(row["person_id"]): row
        for _, row in attribute_dataframe.iterrows()
    }

    image_paths = sorted(image_dir.glob("*.jpg"))

    if max_images is not None:
        image_paths = image_paths[:max_images]

    rows = []
    missing_attributes = 0
    invalid_person_id = 0

    for image_path in image_paths:
        person_id = extract_person_id_from_filename(image_path)

        if person_id is None:
            invalid_person_id += 1
            continue

        attribute_row = attributes_by_person_id.get(person_id)

        if attribute_row is None:
            missing_attributes += 1
            continue

        labels = build_labels_from_attributes(attribute_row)

        rows.append(
            build_output_row(
                image_path=image_path,
                labels=labels,
                source_split=split,
                person_id=person_id,
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output_dataframe = pd.DataFrame(rows)
    output_dataframe.to_csv(output_csv, index=False)

    print("\nDataset Market-1501 Attribute criado.")
    print("CSV:", output_csv)
    print("Split:", split)
    print("Total de imagens:", len(output_dataframe))
    print("IDs inválidos ignorados:", invalid_person_id)
    print("Imagens sem atributos ignoradas:", missing_attributes)
    print("Relatório de mapeamento:", output_report)

    if len(output_dataframe) > 0:
        metadata_cols = [
            "frame_path",
            "source_dataset",
            "source_split",
            "source_person_id",
        ]

        label_cols = [
            column for column in output_dataframe.columns
            if column not in metadata_cols
        ]

        counts = output_dataframe[label_cols].sum().sort_values(ascending=False)

        print("\nDistribuição por label:")
        for label, count in counts.items():
            if int(count) > 0:
                print("{}: {}".format(label, int(count)))

        zero_labels = [
            label for label, count in counts.items()
            if int(count) == 0
        ]

        print("\nLabels sem exemplos:")
        print(zero_labels)

        print("\nMédia de labels positivas por imagem:")
        print(output_dataframe[label_cols].sum(axis=1).mean())

    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Cria dataset para SpectraPersonNet usando Market-1501 Attribute."
    )

    parser.add_argument(
        "--market-root-dir",
        required=True,
        help="Pasta Market-1501-v15.09.15 contendo bounding_box_train/test/query.",
    )

    parser.add_argument(
        "--market-attribute-path",
        required=True,
        help="Caminho para market_attribute.mat.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "test", "query"],
        default="train",
        help="Split de imagens usado.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limite de imagens para debug.",
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/spectra_person_labels.csv",
        help="CSV final no formato da Spectra.",
    )

    parser.add_argument(
        "--output-report",
        default="data/datasets/spectra_person_market1501_mapping_report.json",
        help="Relatório JSON de mapeamento de atributos.",
    )

    args = parser.parse_args()

    build_market1501_person_dataset(
        market_root_dir=args.market_root_dir,
        market_attribute_path=args.market_attribute_path,
        output_csv=args.output_csv,
        split=args.split,
        max_images=args.max_images,
        output_report=args.output_report,
    )


if __name__ == "__main__":
    main()