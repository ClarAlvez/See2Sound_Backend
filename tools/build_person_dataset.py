import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ai.spectra.Person.labels import LABELS


FOCUS_LABELS = [
    "child",
    "adult",
    "elderly",

    "glasses",
    "hat",
    "dress",
    "bald_hair",

    "black_hair",
    "blonde_hair",
    "brown_hair",
    "gray_hair",

    "orange_clothes",
    "pink_clothes",
    "purple_clothes",
    "brown_clothes",
    "gray_clothes",

    "bag",
    "backpack",
]


def read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    return pd.read_csv(csv_path, low_memory=False)


def ensure_person_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "frame_path" not in df.columns:
        raise ValueError("Dataset precisa ter coluna frame_path.")

    for label in LABELS:
        if label not in df.columns:
            df[label] = 0

        df[label] = pd.to_numeric(
            df[label],
            errors="coerce",
        ).fillna(0).astype(int)

        df[label] = df[label].clip(0, 1)

    df["person"] = 1

    return df


def clean_frame_paths(
    df: pd.DataFrame,
    check_images: bool = True,
) -> pd.DataFrame:
    df["frame_path"] = df["frame_path"].astype("string")
    df["frame_path"] = df["frame_path"].str.replace("\\", "/", regex=False)

    before = len(df)

    df = df.dropna(subset=["frame_path"])
    df = df[df["frame_path"].str.strip() != ""]
    df = df[df["frame_path"].str.lower() != "nan"]

    after_empty = len(df)

    if before != after_empty:
        print("Linhas removidas por frame_path vazio:", before - after_empty)

    if check_images:
        exists_mask = df["frame_path"].apply(
            lambda value: Path(str(value)).exists()
        )

        missing_count = int((~exists_mask).sum())

        if missing_count > 0:
            print("Imagens ausentes removidas:", missing_count)

            missing_examples = df.loc[~exists_mask, "frame_path"].head(10)

            print("Exemplos ausentes:")
            for path in missing_examples:
                print("-", path)

        df = df[exists_mask].copy()

    return df


def keep_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        column
        for column in df.columns
        if column.startswith("source_")
        or column.startswith("bbox_")
        or column in [
            "detector_confidence",
            "source_dataset",
            "source_split",
            "source_image_name",
            "source_frame_path",
        ]
    ]

    metadata_cols = list(dict.fromkeys(metadata_cols))

    keep_cols = ["frame_path"] + metadata_cols + LABELS

    existing_keep_cols = [
        column
        for column in keep_cols
        if column in df.columns
    ]

    return df[existing_keep_cols]


def clean_dataset(
    df: pd.DataFrame,
    check_images: bool = True,
) -> pd.DataFrame:
    df = ensure_person_columns(df)
    df = clean_frame_paths(df, check_images=check_images)
    df = ensure_person_columns(df)
    df = keep_relevant_columns(df)

    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    df = ensure_person_columns(df)

    print("\nTotal de linhas:", len(df))

    counts = df[LABELS].sum().sort_values(ascending=False)

    print("\nDistribuição por label:")
    print(counts)

    print("\nLabels zeradas:")
    print(list(counts[counts == 0].index))

    print("\nMédia de labels positivas por imagem:")
    print(df[LABELS].sum(axis=1).mean())

    print("\nDistribuição de labels positivas por imagem:")
    print(df[LABELS].sum(axis=1).describe())


def command_clean(args) -> None:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    df = read_csv(input_csv)
    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nCSV limpo salvo em:")
    print(output_csv)

    print_dataset_summary(df)


def command_validate(args) -> None:
    csv_path = Path(args.csv)

    df = read_csv(csv_path)
    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    print("\nDataset válido:")
    print(csv_path)

    print_dataset_summary(df)


def command_merge(args) -> None:
    input_paths = [
        Path(path)
        for path in args.inputs
    ]

    frames = []

    for path in input_paths:
        print("Lendo:", path)

        df = read_csv(path)
        df = clean_dataset(
            df,
            check_images=not args.no_check_images,
        )

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    before = len(merged)

    if not args.keep_duplicates:
        merged = merged.drop_duplicates(subset=["frame_path"])

    after = len(merged)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("\nDataset unido salvo em:")
    print(output_csv)

    print("Linhas antes:", before)
    print("Linhas depois:", after)

    print_dataset_summary(merged)


def command_balance(args) -> None:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    df = read_csv(input_csv)
    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    selected_parts = []

    print("\nSelecionando exemplos de labels foco:")

    for label in FOCUS_LABELS:
        if label not in df.columns:
            print(f"- {label}: coluna ausente")
            continue

        positives = df[df[label] == 1]

        if positives.empty:
            print(f"- {label}: 0 exemplos")
            continue

        sample_size = min(args.per_focus_label, len(positives))

        sampled = positives.sample(
            n=sample_size,
            random_state=args.random_state,
            replace=False,
        )

        selected_parts.append(sampled)

        print(f"- {label}: {sample_size}")

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
        selected = selected.drop_duplicates(subset=["frame_path"])
    else:
        selected = pd.DataFrame(columns=df.columns)

    remaining_needed = args.target_total - len(selected)

    if remaining_needed > 0:
        remaining_pool = df[~df["frame_path"].isin(selected["frame_path"])]

        random_count = min(remaining_needed, len(remaining_pool))

        if random_count > 0:
            random_part = remaining_pool.sample(
                n=random_count,
                random_state=args.random_state,
                replace=False,
            )

            selected = pd.concat([selected, random_part], ignore_index=True)

    selected = selected.drop_duplicates(subset=["frame_path"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)

    print("\nDataset balanceado salvo em:")
    print(output_csv)

    print_dataset_summary(selected)


def command_real_crops(args) -> None:
    base_csv = Path(args.base_csv)
    real_crops_csv = Path(args.real_crops_csv)
    output_csv = Path(args.output_csv)

    base_df = clean_dataset(
        read_csv(base_csv),
        check_images=not args.no_check_images,
    )

    real_df = clean_dataset(
        read_csv(real_crops_csv),
        check_images=not args.no_check_images,
    )

    parts = [base_df]

    for _ in range(args.repeat):
        parts.append(real_df)

    merged = pd.concat(parts, ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("\nDataset com crops reais criado:")
    print(output_csv)

    print("Base:", base_csv)
    print("Crops reais:", real_crops_csv)
    print("Repetições:", args.repeat)

    print_dataset_summary(merged)


def command_sample(args) -> None:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    df = read_csv(input_csv)
    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    sample_size = min(args.total, len(df))

    sampled = df.sample(
        n=sample_size,
        random_state=args.random_state,
        replace=False,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_csv, index=False)

    print("\nAmostra salva em:")
    print(output_csv)

    print_dataset_summary(sampled)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Builder geral de datasets da SpectraPersonNet."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--input-csv", required=True)
    clean_parser.add_argument("--output-csv", required=True)
    clean_parser.add_argument("--no-check-images", action="store_true")
    clean_parser.set_defaults(func=command_clean)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--csv", required=True)
    validate_parser.add_argument("--no-check-images", action="store_true")
    validate_parser.set_defaults(func=command_validate)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", nargs="+", required=True)
    merge_parser.add_argument("--output-csv", required=True)
    merge_parser.add_argument("--keep-duplicates", action="store_true")
    merge_parser.add_argument("--no-check-images", action="store_true")
    merge_parser.set_defaults(func=command_merge)

    balance_parser = subparsers.add_parser("balance")
    balance_parser.add_argument("--input-csv", required=True)
    balance_parser.add_argument("--output-csv", required=True)
    balance_parser.add_argument("--target-total", type=int, default=30000)
    balance_parser.add_argument("--per-focus-label", type=int, default=2500)
    balance_parser.add_argument("--random-state", type=int, default=42)
    balance_parser.add_argument("--no-check-images", action="store_true")
    balance_parser.set_defaults(func=command_balance)

    real_parser = subparsers.add_parser("real-crops")
    real_parser.add_argument("--base-csv", required=True)
    real_parser.add_argument("--real-crops-csv", required=True)
    real_parser.add_argument("--output-csv", required=True)
    real_parser.add_argument("--repeat", type=int, default=3)
    real_parser.add_argument("--no-check-images", action="store_true")
    real_parser.set_defaults(func=command_real_crops)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--input-csv", required=True)
    sample_parser.add_argument("--output-csv", required=True)
    sample_parser.add_argument("--total", type=int, default=300)
    sample_parser.add_argument("--random-state", type=int, default=42)
    sample_parser.add_argument("--no-check-images", action="store_true")
    sample_parser.set_defaults(func=command_sample)
    
    utk_parser = subparsers.add_parser("from-utkface")
    utk_parser.add_argument("--input-dir", required=True)
    utk_parser.add_argument("--output-csv", required=True)
    utk_parser.add_argument("--max-rows", type=int, default=None)
    utk_parser.add_argument("--random-state", type=int, default=42)
    utk_parser.add_argument("--no-check-images", action="store_true")
    utk_parser.set_defaults(func=command_from_utkface)

    return parser

def age_to_spectra_labels(age: int):
    if age <= 12:
        return {
            "child": 1,
            "adult": 0,
            "elderly": 0,
        }

    if age >= 60:
        return {
            "child": 0,
            "adult": 0,
            "elderly": 1,
        }

    return {
        "child": 0,
        "adult": 1,
        "elderly": 0,
    }
    
def command_from_utkface(args) -> None:
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {input_dir}")

    image_paths = (
        list(input_dir.rglob("*.jpg"))
        + list(input_dir.rglob("*.jpeg"))
        + list(input_dir.rglob("*.png"))
    )

    rows = []
    skipped = 0

    for image_path in sorted(image_paths):
        name = image_path.name

        try:
            age_text = name.split("_")[0]
            age = int(age_text)
        except Exception:
            skipped += 1
            continue

        labels = {
            label: 0
            for label in LABELS
        }

        labels["person"] = 1

        age_labels = age_to_spectra_labels(age)

        for label, value in age_labels.items():
            if label in labels:
                labels[label] = value

        row = {
            "frame_path": str(image_path),
            "source_dataset": "utkface",
            "source_age": age,
        }

        row.update(labels)
        rows.append(row)

    df = pd.DataFrame(rows)

    if args.max_rows is not None:
        df = df.sample(
            n=min(args.max_rows, len(df)),
            random_state=args.random_state,
            replace=False,
        )

    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nDataset UTKFace convertido:")
    print(output_csv)
    print("Linhas:", len(df))
    print("Ignoradas:", skipped)

    print_dataset_summary(df)

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()