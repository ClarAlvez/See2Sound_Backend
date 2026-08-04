import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ai.spectra.Person.labels import SPECTRA_PERSON_LABELS


CELEBA_TO_SPECTRA = {
    "Male": {
        1: "man",
        -1: "woman",
    },

    "Bald": "bald_hair",
    "Black_Hair": "black_hair",
    "Blond_Hair": "blonde_hair",
    "Brown_Hair": "brown_hair",
    "Gray_Hair": "gray_hair",
    "Straight_Hair": "straight_hair",
    "Wavy_Hair": "wavy_hair",
    "Bangs": "bangs_hair",
    "Receding_Hairline": "receding_hairline",

    "Eyeglasses": "glasses",
    "Wearing_Hat": "hat",
}


def is_positive(value) -> bool:
    try:
        return int(value) == 1
    except Exception:
        return False


def read_celeba_attributes(attr_path: Path) -> pd.DataFrame:
    if not attr_path.exists():
        raise FileNotFoundError(f"Arquivo de atributos não encontrado: {attr_path}")

    df = pd.read_csv(
        attr_path,
        sep=r"\s+",
        skiprows=1,
        engine="python",
    )

    # O índice é o nome da imagem: 000001.jpg, 000002.jpg...
    df.index = df.index.astype(str)

    return df


def build_labels_from_celeba_row(row: pd.Series) -> Dict[str, int]:
    labels = {
        label: 0
        for label in SPECTRA_PERSON_LABELS
    }

    if "person" in labels:
        labels["person"] = 1

    # Gênero no CelebA:
    # Male = 1 -> man
    # Male = -1 -> woman
    if "Male" in row.index:
        try:
            male_value = int(row["Male"])

            if male_value == 1 and "man" in labels:
                labels["man"] = 1

            if male_value == -1 and "woman" in labels:
                labels["woman"] = 1

        except Exception:
            pass

    for celeba_attr, spectra_label in CELEBA_TO_SPECTRA.items():
        if celeba_attr == "Male":
            continue

        if celeba_attr not in row.index:
            continue

        if spectra_label not in labels:
            continue

        if is_positive(row[celeba_attr]):
            labels[spectra_label] = 1

    apply_consistency_rules(labels)

    return labels


def apply_consistency_rules(labels: Dict[str, int]) -> None:
    # Evita man e woman simultâneos.
    if labels.get("man", 0) == 1 and labels.get("woman", 0) == 1:
        labels["woman"] = 0

    # Bald deve sobrepor comprimento/textura.
    if labels.get("bald_hair", 0) == 1:
        for label in [
            "short_hair",
            "long_hair",
            "black_hair",
            "blonde_hair",
            "brown_hair",
            "gray_hair",
            "straight_hair",
            "wavy_hair",
            "bangs_hair",
        ]:
            if label in labels:
                labels[label] = 0

    # Cores de cabelo: se mais de uma cor aparecer, mantém uma prioridade simples.
    hair_color_labels = [
        "black_hair",
        "blonde_hair",
        "brown_hair",
        "gray_hair",
    ]

    active_colors = [
        label
        for label in hair_color_labels
        if labels.get(label, 0) == 1
    ]

    if len(active_colors) > 1:
        priority = [
            "gray_hair",
            "black_hair",
            "brown_hair",
            "blonde_hair",
        ]

        chosen = None

        for label in priority:
            if label in active_colors:
                chosen = label
                break

        for label in hair_color_labels:
            labels[label] = 1 if label == chosen else 0

    # Textura: se straight e wavy vierem juntos, mantém wavy.
    if labels.get("straight_hair", 0) == 1 and labels.get("wavy_hair", 0) == 1:
        labels["straight_hair"] = 0


def build_celeba_dataset(
    celeba_root: Path,
    output_csv_path: Path,
    output_report_path: Path,
    max_rows: Optional[int] = None,
) -> None:
    attr_path = celeba_root / "list_attr_celeba.txt"
    images_dir = celeba_root / "img_align_celeba"

    if not images_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {images_dir}")

    df = read_celeba_attributes(attr_path)

    if max_rows is not None:
        df = df.head(max_rows)

    rows = []
    missing_images = 0
    missing_samples = []

    for image_name, row in df.iterrows():
        image_path = images_dir / image_name

        if not image_path.exists():
            missing_images += 1

            if len(missing_samples) < 20:
                missing_samples.append(str(image_path))

            continue

        labels = build_labels_from_celeba_row(row)

        output_row = {
            "frame_path": str(image_path),
            "source_dataset": "celeba",
            "source_image_name": image_name,
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
        "celeba_root": str(celeba_root),
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

    print("\nDataset CelebA convertido com sucesso.")
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
        description="Converte CelebA para o formato da SpectraPersonNet."
    )

    parser.add_argument(
        "--celeba-root",
        default="data/external/celeba/celeba",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
    )

    parser.add_argument(
        "--output-report",
        required=True,
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    build_celeba_dataset(
        celeba_root=Path(args.celeba_root),
        output_csv_path=Path(args.output_csv),
        output_report_path=Path(args.output_report),
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
