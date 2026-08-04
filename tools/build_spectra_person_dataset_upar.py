import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ai.spectra.Person.labels import SPECTRA_PERSON_LABELS


def normalize_name(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("#", "")
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = value.replace("/", "_")
    value = value.replace("\\", "_")
    value = value.replace("&", "_")
    value = value.replace("(", "_")
    value = value.replace(")", "_")
    value = value.replace(".", "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")


def is_positive(value) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    try:
        return float(value) >= 0.5
    except Exception:
        pass

    text = str(value).strip().lower()

    return text in {
        "1",
        "true",
        "yes",
        "y",
        "positive",
        "present",
        "sim",
    }


def read_upar_csv(csv_path: Path) -> pd.DataFrame:
    """
    O train.csv do UPAR geralmente vem com o caminho da imagem como índice.
    Por isso tentamos ler com index_col=0 primeiro.
    """
    df = pd.read_csv(csv_path, index_col=0)

    # Se por algum motivo o índice não parecer caminho de imagem,
    # tenta leitura normal.
    sample_index = str(df.index[0]) if len(df) > 0 else ""

    if "/" not in sample_index and "\\" not in sample_index:
        df = pd.read_csv(csv_path)

    return df


def get_image_reference(row_index, row: pd.Series) -> str:
    """
    No UPAR, o caminho da imagem costuma estar no índice:
    Market1501/bounding_box_train/0002_c1s1_000451_03.jpg
    """
    index_text = str(row_index)

    if index_text and index_text.lower() != "nan":
        if "/" in index_text or "\\" in index_text:
            return index_text

    image_column_candidates = [
        "image_path",
        "img_path",
        "path",
        "file_path",
        "filepath",
        "filename",
        "file_name",
        "image",
        "img",
        "name",
        "Unnamed: 0",
    ]

    normalized_columns = {
        normalize_name(column): column
        for column in row.index
    }

    for candidate in image_column_candidates:
        normalized_candidate = normalize_name(candidate)

        if normalized_candidate in normalized_columns:
            column = normalized_columns[normalized_candidate]
            return str(row[column])

    raise ValueError(
        "Não encontrei referência de imagem nem no índice nem nas colunas."
    )


def resolve_image_path(image_reference: str, image_root_dir: Path) -> Optional[Path]:
    image_reference = str(image_reference).strip().replace("\\", "/")

    if not image_reference:
        return None

    direct_path = Path(image_reference)

    if direct_path.exists():
        return direct_path

    candidate = image_root_dir / image_reference

    if candidate.exists():
        return candidate

    # Alguns CSVs podem vir com caminhos começando com ./.
    cleaned_reference = image_reference.lstrip("./")
    candidate = image_root_dir / cleaned_reference

    if candidate.exists():
        return candidate

    # Fallback: procura pelo nome do arquivo dentro da raiz.
    filename = Path(image_reference).name
    matches = list(image_root_dir.rglob(filename))

    if matches:
        return matches[0]

    return None


def get_value(row: pd.Series, normalized_columns: Dict[str, str], candidates: List[str]):
    for candidate in candidates:
        normalized = normalize_name(candidate)

        if normalized in normalized_columns:
            return row[normalized_columns[normalized]]

    return None


UPAR_TO_SPECTRA_CANDIDATES = {
    "woman": [
        "Gender-Female",
        "gender_female",
    ],

    "short_hair": [
        "Hair-Length-Short",
        "hair_length_short",
    ],

    "long_hair": [
        "Hair-Length-Long",
        "hair_length_long",
    ],

    "bald_hair": [
        "Hair-Length-Bald",
        "hair_length_bald",
    ],

    "black_clothes": [
        "UpperBody-Color-Black",
        "LowerBody-Color-Black",
    ],

    "white_clothes": [
        "UpperBody-Color-White",
        "LowerBody-Color-White",
    ],

    "red_clothes": [
        "UpperBody-Color-Red",
        "LowerBody-Color-Red",
    ],

    "blue_clothes": [
        "UpperBody-Color-Blue",
        "LowerBody-Color-Blue",
    ],

    "green_clothes": [
        "UpperBody-Color-Green",
        "LowerBody-Color-Green",
    ],

    "yellow_clothes": [
        "UpperBody-Color-Yellow",
        "LowerBody-Color-Yellow",
    ],

    "brown_clothes": [
        "UpperBody-Color-Brown",
        "LowerBody-Color-Brown",
    ],

    "gray_clothes": [
        "UpperBody-Color-Grey",
        "LowerBody-Color-Grey",
        "UpperBody-Color-Gray",
        "LowerBody-Color-Gray",
    ],

    "orange_clothes": [
        "UpperBody-Color-Orange",
        "LowerBody-Color-Orange",
    ],

    "pink_clothes": [
        "UpperBody-Color-Pink",
        "LowerBody-Color-Pink",
    ],

    "purple_clothes": [
        "UpperBody-Color-Purple",
        "LowerBody-Color-Purple",
    ],

    "dress": [
        "LowerBody-Type-Skirt&Dress",
        "LowerBody-Type-Skirt-Dress",
        "LowerBody-Type-Skirt_Dress",
    ],

    "backpack": [
        "Accessory-Backpack",
    ],

    "bag": [
        "Accessory-Bag",
    ],

    "glasses": [
        "Accessory-Glasses-Normal",
        "Accessory-Glasses-Sun",
    ],

    "hat": [
        "Accessory-Hat",
    ],
}


def apply_consistency_rules(labels: Dict[str, int]) -> None:
    if "man" in labels and "woman" in labels:
        if labels["man"] == 1 and labels["woman"] == 1:
            labels["man"] = 0

    hair_group = [
        label
        for label in ["short_hair", "long_hair", "bald_hair"]
        if label in labels
    ]

    active_hair = [
        label
        for label in hair_group
        if labels.get(label, 0) == 1
    ]

    if len(active_hair) > 1:
        priority = ["bald_hair", "long_hair", "short_hair"]

        chosen = None

        for label in priority:
            if label in active_hair:
                chosen = label
                break

        for label in hair_group:
            labels[label] = 1 if label == chosen else 0


def build_labels_from_upar_row(row: pd.Series) -> Dict[str, int]:
    labels = {
        label: 0
        for label in SPECTRA_PERSON_LABELS
    }

    if "person" in labels:
        labels["person"] = 1

    normalized_columns = {
        normalize_name(column): column
        for column in row.index
    }

    female_value = get_value(
        row=row,
        normalized_columns=normalized_columns,
        candidates=[
            "Gender-Female",
            "gender_female",
        ],
    )

    if female_value is not None:
        if is_positive(female_value):
            if "woman" in labels:
                labels["woman"] = 1
        else:
            if "man" in labels:
                labels["man"] = 1

    for spectra_label, candidates in UPAR_TO_SPECTRA_CANDIDATES.items():
        if spectra_label not in labels:
            continue

        if spectra_label == "woman":
            continue

        for candidate in candidates:
            value = get_value(
                row=row,
                normalized_columns=normalized_columns,
                candidates=[candidate],
            )

            if value is not None and is_positive(value):
                labels[spectra_label] = 1
                break

    apply_consistency_rules(labels)

    return labels


def build_upar_dataset(
    annotations_csv_path: Path,
    image_root_dir: Path,
    output_csv_path: Path,
    output_report_path: Path,
    max_rows: Optional[int] = None,
    source_name: str = "upar",
) -> None:
    if not annotations_csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {annotations_csv_path}")

    if not image_root_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {image_root_dir}")

    df = read_upar_csv(annotations_csv_path)

    if max_rows is not None:
        df = df.head(max_rows)

    rows = []
    missing_images = 0
    missing_samples = []

    for index, row in df.iterrows():
        image_reference = get_image_reference(index, row)

        image_path = resolve_image_path(
            image_reference=image_reference,
            image_root_dir=image_root_dir,
        )

        if image_path is None:
            missing_images += 1

            if len(missing_samples) < 20:
                missing_samples.append(image_reference)

            continue

        labels = build_labels_from_upar_row(row)

        output_row = {
            "frame_path": str(image_path),
            "source_dataset": source_name,
            "source_annotation_csv": str(annotations_csv_path),
            "source_image_reference": image_reference,
        }

        for label in SPECTRA_PERSON_LABELS:
            output_row[label] = int(labels.get(label, 0))

        rows.append(output_row)

    output_df = pd.DataFrame(rows)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_csv_path, index=False)

    label_counts = {}

    if len(output_df) > 0:
        label_counts = {
            label: int(output_df[label].sum())
            for label in SPECTRA_PERSON_LABELS
            if label in output_df.columns
        }

    report = {
        "annotations_csv_path": str(annotations_csv_path),
        "image_root_dir": str(image_root_dir),
        "total_input_rows": int(len(df)),
        "total_output_rows": int(len(output_df)),
        "missing_images": int(missing_images),
        "missing_image_samples": missing_samples,
        "spectra_person_labels": SPECTRA_PERSON_LABELS,
        "available_columns": list(df.columns),
        "label_counts": label_counts,
    }

    with open(output_report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\nDataset UPAR convertido com sucesso.")
    print("CSV:", output_csv_path)
    print("Relatório:", output_report_path)
    print("Linhas de entrada:", len(df))
    print("Linhas de saída:", len(output_df))
    print("Imagens ausentes:", missing_images)

    if len(output_df) > 0:
        counts = output_df[SPECTRA_PERSON_LABELS].sum().sort_values(ascending=False)

        print("\nDistribuição por label:")
        print(counts)

        print("\nLabels zeradas:")
        print(list(counts[counts == 0].index))

        print("\nMédia de labels positivas por imagem:")
        print(output_df[SPECTRA_PERSON_LABELS].sum(axis=1).mean())


def main():
    parser = argparse.ArgumentParser(
        description="Converte UPAR para o formato da SpectraPersonNet."
    )

    parser.add_argument(
        "--annotations-csv",
        required=True,
        help="CSV de anotações do UPAR.",
    )

    parser.add_argument(
        "--image-root-dir",
        required=True,
        help="Pasta raiz onde estão Market1501, PA100k e PETA.",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="CSV final da Spectra.",
    )

    parser.add_argument(
        "--output-report",
        required=True,
        help="Relatório JSON.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--source-name",
        default="upar",
    )

    args = parser.parse_args()

    build_upar_dataset(
        annotations_csv_path=Path(args.annotations_csv),
        image_root_dir=Path(args.image_root_dir),
        output_csv_path=Path(args.output_csv),
        output_report_path=Path(args.output_report),
        max_rows=args.max_rows,
        source_name=args.source_name,
    )


if __name__ == "__main__":
    main()
