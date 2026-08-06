import argparse
from pathlib import Path
from typing import List

import pandas as pd

from ai.spectra.Person.labels import LABELS


FOCUS_LABELS = [
    "child",
    "elderly",
    "glasses",
    "hat",
    "dress",
    "bald_hair",
    "orange_clothes",
    "pink_clothes",
    "purple_clothes",
    "brown_clothes",
    "gray_clothes",
]


def ensure_person_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "frame_path" not in df.columns:
        raise ValueError("Dataset precisa ter coluna frame_path.")

    for label in LABELS:
        if label not in df.columns:
            df[label] = 0

        df[label] = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
        df[label] = df[label].clip(0, 1)

    df["person"] = 1

    return df


def keep_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        column
        for column in df.columns
        if column.startswith("source_")
        or column.startswith("bbox_")
        or column in ["detector_confidence"]
    ]

    keep_cols = ["frame_path"] + metadata_cols + LABELS

    return df[keep_cols]


def clean_dataset(df: pd.DataFrame, check_images: bool = True) -> pd.DataFrame:
    df = ensure_person_columns(df)

    df["frame_path"] = df["frame_path"].astype("string")
    df = df.dropna(subset=["frame_path"])
    df = df[df["frame_path"].str.strip() != ""]
    df = df[df["frame_path"].str.lower() != "nan"]

    if check_images:
        exists_mask = df["frame_path"].apply(lambda value: Path(str(value)).exists())
        missing_count = int((~exists_mask).sum())

        if missing_count > 0:
            print("Imagens ausentes removidas:", missing_count)

        df = df[exists_mask].copy()

    return keep_relevant_columns(df)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")

    return pd.read_csv(path, low_memory=False)


def merge_datasets(input_paths: List[Path], output_csv: Path, check_images: bool = True) -> None:
    frames = []

    for path in input_paths:
        print("Lendo:", path)
        df = read_csv(path)
        df = clean_dataset(df, check_images=check_images)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    before = len(merged)
    merged = merged.drop_duplicates(subset=["frame_path"])
    after = len(merged)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("\nDataset unido salvo em:", output_csv)
    print("Linhas antes:", before)
    print("Linhas depois de remover duplicados:", after)

    print_dataset_summary(merged)


def create_balanced_subset(
    input_csv: Path,
    output_csv: Path,
    target_total: int,
    per_focus_label: int,
    random_state: int,
    check_images: bool = True,
) -> None:
    df = read_csv(input_csv)
    df = clean_dataset(df, check_images=check_images)

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

        sample_size = min(per_focus_label, len(positives))

        sampled = positives.sample(
            n=sample_size,
            random_state=random_state,
            replace=False,
        )

        selected_parts.append(sampled)
        print(f"- {label}: {sample_size}")

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
        selected = selected.drop_duplicates(subset=["frame_path"])
    else:
        selected = pd.DataFrame(columns=df.columns)

    remaining_needed = target_total - len(selected)

    if remaining_needed > 0:
        remaining_pool = df[~df["frame_path"].isin(selected["frame_path"])]

        random_part = remaining_pool.sample(
            n=min(remaining_needed, len(remaining_pool)),
            random_state=random_state,
            replace=False,
        )

        selected = pd.concat([selected, random_part], ignore_index=True)

    selected = selected.drop_duplicates(subset=["frame_path"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)

    print("\nSubset balanceado salvo em:", output_csv)
    print("Total final:", len(selected))

    print_dataset_summary(selected)


def repeat_real_crops(base_csv: Path, real_crops_csv: Path, output_csv: Path, repeat: int) -> None:
    base_df = clean_dataset(read_csv(base_csv), check_images=True)
    real_df = clean_dataset(read_csv(real_crops_csv), check_images=True)

    parts = [base_df]

    for _ in range(repeat):
        parts.append(real_df)

    merged = pd.concat(parts, ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("\nDataset com crops reais criado:", output_csv)
    print("Repetições dos crops reais:", repeat)
    print("Total:", len(merged))

    print_dataset_summary(merged)


def print_dataset_summary(df: pd.DataFrame) -> None:
    df = ensure_person_columns(df)

    counts = df[LABELS].sum().sort_values(ascending=False)

    print("\nDistribuição por label:")
    print(counts)

    print("\nLabels zeradas:")
    print(list(counts[counts == 0].index))

    print("\nMédia de labels positivas por imagem:")
    print(df[LABELS].sum(axis=1).mean())


def validate_dataset(csv_path: Path, check_images: bool = True) -> None:
    df = read_csv(csv_path)
    df = clean_dataset(df, check_images=check_images)

    print("Arquivo:", csv_path)
    print("Total de linhas válidas:", len(df))

    if check_images:
        print("Imagens verificadas.")

    print_dataset_summary(df)


def main():
    parser = argparse.ArgumentParser(
        description="Builder geral de datasets da SpectraPersonNet."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", nargs="+", required=True)
    merge_parser.add_argument("--output-csv", required=True)
    merge_parser.add_argument("--no-check-images", action="store_true")

    balance_parser = subparsers.add_parser("balance")
    balance_parser.add_argument("--input-csv", required=True)
    balance_parser.add_argument("--output-csv", required=True)
    balance_parser.add_argument("--target-total", type=int, default=30000)
    balance_parser.add_argument("--per-focus-label", type=int, default=2500)
    balance_parser.add_argument("--random-state", type=int, default=42)
    balance_parser.add_argument("--no-check-images", action="store_true")

    real_parser = subparsers.add_parser("real-crops")
    real_parser.add_argument("--base-csv", required=True)
    real_parser.add_argument("--real-crops-csv", required=True)
    real_parser.add_argument("--output-csv", required=True)
    real_parser.add_argument("--repeat", type=int, default=3)

    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--input-csv", required=True)
    clean_parser.add_argument("--output-csv", required=True)
    clean_parser.add_argument("--no-check-images", action="store_true")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--csv", required=True)
    validate_parser.add_argument("--no-check-images", action="store_true")

    args = parser.parse_args()

    if args.command == "merge":
        merge_datasets(
            input_paths=[Path(path) for path in args.inputs],
            output_csv=Path(args.output_csv),
            check_images=not args.no_check_images,
        )

    elif args.command == "balance":
        create_balanced_subset(
            input_csv=Path(args.input_csv),
            output_csv=Path(args.output_csv),
            target_total=args.target_total,
            per_focus_label=args.per_focus_label,
            random_state=args.random_state,
            check_images=not args.no_check_images,
        )

    elif args.command == "real-crops":
        repeat_real_crops(
            base_csv=Path(args.base_csv),
            real_crops_csv=Path(args.real_crops_csv),
            output_csv=Path(args.output_csv),
            repeat=args.repeat,
        )

    elif args.command == "clean":
        df = read_csv(Path(args.input_csv))
        df = clean_dataset(df, check_images=not args.no_check_images)

        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

        print("\nCSV limpo salvo em:", output_csv)
        print_dataset_summary(df)

    elif args.command == "validate":
        validate_dataset(
            csv_path=Path(args.csv),
            check_images=not args.no_check_images,
        )


if __name__ == "__main__":
    main()