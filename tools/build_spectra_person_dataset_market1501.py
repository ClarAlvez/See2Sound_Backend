import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from scipy.io import loadmat

from ai.spectra.labels.label_sets import SPECTRA_PERSON_LABELS


def extract_person_id_from_filename(image_path: Path) -> Optional[str]:
    """
    Exemplo:
    0002_c1s1_000451_03.jpg -> 0002
    """
    first_part = image_path.name.split("_")[0]

    if not first_part.isdigit():
        return None

    if int(first_part) < 0:
        return None

    return first_part.zfill(4)


def read_market_split(market_attribute_path: Path, split_name: str) -> pd.DataFrame:
    data = loadmat(str(market_attribute_path))

    if "market_attribute" not in data:
        raise ValueError("O arquivo .mat não possui a chave 'market_attribute'.")

    market = data["market_attribute"][0, 0]

    if split_name not in market.dtype.names:
        raise ValueError(f"Split '{split_name}' não encontrado no market_attribute.mat.")

    split = market[split_name][0, 0]

    rows = {}

    for field in split.dtype.names:
        values = split[field].squeeze()

        if field == "image_index":
            parsed_values = []

            for item in values:
                try:
                    parsed_values.append(str(item[0]).strip().zfill(4))
                except Exception:
                    parsed_values.append(str(item).strip().zfill(4))

            rows[field] = parsed_values

        else:
            rows[field] = [int(value) for value in values]

    dataframe = pd.DataFrame(rows)

    if "image_index" not in dataframe.columns:
        raise ValueError("O split não possui campo image_index.")

    dataframe["person_id"] = dataframe["image_index"].astype(str).str.zfill(4)

    return dataframe


def is_positive(value) -> bool:
    """
    No Market-1501 Attribute:
    1 = ausência / não
    2 = presença / sim
    """
    try:
        return int(value) == 2
    except Exception:
        return False


def build_labels_from_market_row(row: pd.Series) -> Dict[str, int]:
    labels = {label: 0 for label in SPECTRA_PERSON_LABELS}

    if "person" in labels:
        labels["person"] = 1

    # gender:
    # 1 = man
    # 2 = woman
    if "gender" in row.index:
        gender = int(row["gender"])

        if gender == 1 and "man" in labels:
            labels["man"] = 1

        if gender == 2 and "woman" in labels:
            labels["woman"] = 1

    # hair:
    # 1 = short_hair
    # 2 = long_hair
    if "hair" in row.index:
        hair = int(row["hair"])

        if hair == 1 and "short_hair" in labels:
            labels["short_hair"] = 1

        if hair == 2 and "long_hair" in labels:
            labels["long_hair"] = 1

    mapping = {
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

    for market_label, spectra_label in mapping.items():
        if market_label not in row.index:
            continue

        if spectra_label not in labels:
            continue

        if is_positive(row[market_label]):
            labels[spectra_label] = 1

    return labels


def build_dataset(
    market_root_dir: Path,
    market_attribute_path: Path,
    split: str,
    output_csv: Path,
    output_report: Path,
    max_images: Optional[int] = None,
):
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
        raise ValueError("Split inválido. Use train, test ou query.")

    if not image_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {image_dir}")

    if not market_attribute_path.exists():
        raise FileNotFoundError(f"Arquivo .mat não encontrado: {market_attribute_path}")

    attribute_df = read_market_split(
        market_attribute_path=market_attribute_path,
        split_name=attribute_split,
    )

    attributes_by_id = {
        str(row["person_id"]).zfill(4): row
        for _, row in attribute_df.iterrows()
    }

    image_paths = sorted(image_dir.glob("*.jpg"))

    if max_images is not None:
        image_paths = image_paths[:max_images]

    rows = []
    missing_attribute_ids = set()
    invalid_images = 0

    for image_path in image_paths:
        person_id = extract_person_id_from_filename(image_path)

        if person_id is None:
            invalid_images += 1
            continue

        attribute_row = attributes_by_id.get(person_id)

        if attribute_row is None:
            missing_attribute_ids.add(person_id)
            continue

        labels = build_labels_from_market_row(attribute_row)

        output_row = {
            "frame_path": str(image_path),
            "source_dataset": "market1501_attribute",
            "source_split": split,
            "source_person_id": person_id,
        }

        for label in SPECTRA_PERSON_LABELS:
            output_row[label] = int(labels.get(label, 0))

        rows.append(output_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_csv, index=False)

    metadata_cols = [
        "frame_path",
        "source_dataset",
        "source_split",
        "source_person_id",
    ]

    label_cols = [
        column for column in output_df.columns
        if column not in metadata_cols
    ]

    label_counts = {}

    if len(output_df) > 0:
        label_counts = {
            label: int(output_df[label].sum())
            for label in label_cols
        }

    report = {
        "split": split,
        "attribute_split": attribute_split,
        "image_dir": str(image_dir),
        "output_csv": str(output_csv),
        "total_rows": int(len(output_df)),
        "invalid_images": int(invalid_images),
        "missing_attribute_ids_count": int(len(missing_attribute_ids)),
        "missing_attribute_ids_sample": sorted(list(missing_attribute_ids))[:50],
        "label_counts": label_counts,
        "spectra_person_labels": SPECTRA_PERSON_LABELS,
    }

    with open(output_report, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\nDataset gerado com sucesso.")
    print("CSV:", output_csv)
    print("Relatório:", output_report)
    print("Total de linhas:", len(output_df))
    print("Imagens inválidas:", invalid_images)
    print("IDs sem atributo:", len(missing_attribute_ids))

    if len(output_df) > 0:
        counts = output_df[label_cols].sum().sort_values(ascending=False)

        print("\nDistribuição por label:")
        print(counts)

        print("\nMédia de labels positivas por imagem:")
        print(output_df[label_cols].sum(axis=1).mean())


def main():
    parser = argparse.ArgumentParser(
        description="Gera dataset da SpectraPersonNet usando Market-1501 Attribute."
    )

    parser.add_argument(
        "--market-root-dir",
        required=True,
        help="Pasta Market-1501-v15.09.15.",
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
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-csv",
        required=True,
    )

    parser.add_argument(
        "--output-report",
        required=True,
    )

    args = parser.parse_args()

    build_dataset(
        market_root_dir=Path(args.market_root_dir),
        market_attribute_path=Path(args.market_attribute_path),
        split=args.split,
        output_csv=Path(args.output_csv),
        output_report=Path(args.output_report),
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()