import argparse
from pathlib import Path
from typing import List
from datasets import load_dataset

import pandas as pd

from ai.spectra.Actions.labels import LABELS


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    return pd.read_csv(csv_path, low_memory=False)


def normalize_path_text(value: str) -> str:
    return str(value).replace("\\", "/")


def ensure_action_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "frame_path" not in df.columns:
        raise ValueError("Dataset precisa ter coluna frame_path.")

    df["frame_path"] = df["frame_path"].astype("string")
    df["frame_path"] = df["frame_path"].str.replace("\\", "/", regex=False)

    for label in LABELS:
        if label not in df.columns:
            df[label] = 0

        df[label] = pd.to_numeric(
            df[label],
            errors="coerce",
        ).fillna(0).astype(int)

        df[label] = df[label].clip(0, 1)

    return df


def clean_frame_paths(
    df: pd.DataFrame,
    check_images: bool = True,
) -> pd.DataFrame:
    before = len(df)

    df = df.dropna(subset=["frame_path"])
    df = df[df["frame_path"].str.strip() != ""]
    df = df[df["frame_path"].str.lower() != "nan"]

    removed_empty = before - len(df)

    if removed_empty > 0:
        print("Linhas removidas por frame_path vazio:", removed_empty)

    if check_images:
        exists_mask = df["frame_path"].apply(
            lambda value: Path(str(value)).exists()
        )

        missing_count = int((~exists_mask).sum())

        if missing_count > 0:
            print("Imagens ausentes removidas:", missing_count)
            print("Exemplos ausentes:")

            for path in df.loc[~exists_mask, "frame_path"].head(10):
                print("-", path)

        df = df[exists_mask].copy()

    return df


def keep_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        column
        for column in df.columns
        if column.startswith("source_")
        or column in [
            "video_path",
            "timestamp",
            "frame_index",
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
    df = ensure_action_columns(df)
    df = clean_frame_paths(df, check_images=check_images)
    df = ensure_action_columns(df)
    df = keep_relevant_columns(df)

    return df


def split_label_folder_name(folder_name: str) -> List[str]:
    folder_name = folder_name.strip()

    if "__" in folder_name:
        parts = folder_name.split("__")
    elif "," in folder_name:
        parts = folder_name.split(",")
    elif ";" in folder_name:
        parts = folder_name.split(";")
    else:
        parts = [folder_name]

    labels = []

    for part in parts:
        label = part.strip()

        if label:
            labels.append(label)

    return labels


def infer_labels_from_folder(
    image_path: Path,
    input_dir: Path,
) -> List[str]:
    relative_path = image_path.relative_to(input_dir)

    if len(relative_path.parts) < 2:
        return []

    label_folder = relative_path.parts[0]

    raw_labels = split_label_folder_name(label_folder)

    valid_labels = []

    for label in raw_labels:
        if label not in LABELS:
            print(f"Label ignorada por não existir em Actions.labels: {label}")
            continue

        valid_labels.append(label)

    return valid_labels


def collect_images(input_dir: Path) -> List[Path]:
    image_paths = []

    for extension in IMAGE_EXTENSIONS:
        image_paths.extend(input_dir.rglob(f"*{extension}"))

    return sorted(image_paths)


def command_from_folders(args) -> None:
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {input_dir}")

    image_paths = collect_images(input_dir)

    rows = []
    skipped = 0

    for image_path in image_paths:
        labels = infer_labels_from_folder(
            image_path=image_path,
            input_dir=input_dir,
        )

        if not labels:
            skipped += 1
            continue

        row = {
            "frame_path": str(image_path),
            "source_dataset": "folder_actions",
        }

        for label in LABELS:
            row[label] = 0

        for label in labels:
            row[label] = 1

        rows.append(row)

    df = pd.DataFrame(rows)

    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nDataset de Actions criado por pastas:")
    print(output_csv)
    print("Imagens encontradas:", len(image_paths))
    print("Linhas salvas:", len(df))
    print("Ignoradas:", skipped)

    print_dataset_summary(df)


def command_from_manifest(args) -> None:
    manifest_csv = Path(args.manifest_csv)
    output_csv = Path(args.output_csv)

    df_manifest = read_csv(manifest_csv)

    if args.image_column not in df_manifest.columns:
        raise ValueError(f"Coluna de imagem não encontrada: {args.image_column}")

    if args.labels_column not in df_manifest.columns:
        raise ValueError(f"Coluna de labels não encontrada: {args.labels_column}")

    rows = []

    for _, source_row in df_manifest.iterrows():
        frame_path = normalize_path_text(source_row[args.image_column])

        raw_labels_text = str(source_row[args.labels_column])
        raw_labels = [
            item.strip()
            for item in raw_labels_text.replace(",", ";").split(";")
            if item.strip()
        ]

        row = {
            "frame_path": frame_path,
            "source_dataset": "manifest_actions",
        }

        for label in LABELS:
            row[label] = 0

        for label in raw_labels:
            if label not in LABELS:
                print(f"Label ignorada por não existir em Actions.labels: {label}")
                continue

            row[label] = 1

        rows.append(row)

    df = pd.DataFrame(rows)

    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nDataset de Actions criado por manifesto:")
    print(output_csv)
    print_dataset_summary(df)


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
    frames = []

    for input_csv in args.inputs:
        path = Path(input_csv)

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


def print_dataset_summary(df: pd.DataFrame) -> None:
    df = ensure_action_columns(df)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Builder geral de datasets da SpectraActionNet."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    folders_parser = subparsers.add_parser("from-folders")
    folders_parser.add_argument("--input-dir", required=True)
    folders_parser.add_argument("--output-csv", required=True)
    folders_parser.add_argument("--no-check-images", action="store_true")
    folders_parser.set_defaults(func=command_from_folders)

    manifest_parser = subparsers.add_parser("from-manifest")
    manifest_parser.add_argument("--manifest-csv", required=True)
    manifest_parser.add_argument("--output-csv", required=True)
    manifest_parser.add_argument("--image-column", default="frame_path")
    manifest_parser.add_argument("--labels-column", default="labels")
    manifest_parser.add_argument("--no-check-images", action="store_true")
    manifest_parser.set_defaults(func=command_from_manifest)

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

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--input-csv", required=True)
    sample_parser.add_argument("--output-csv", required=True)
    sample_parser.add_argument("--total", type=int, default=300)
    sample_parser.add_argument("--random-state", type=int, default=42)
    sample_parser.add_argument("--no-check-images", action="store_true")
    sample_parser.set_defaults(func=command_sample)

    hf_parser = subparsers.add_parser("from-huggingface")
    hf_parser.add_argument(
        "--dataset-name",
        default="Bingsu/Human_Action_Recognition",
    )
    hf_parser.add_argument(
        "--split",
        default="train",
    )
    hf_parser.add_argument(
        "--output-image-dir",
        default="data/external/actions/human_action_recognition",
    )
    hf_parser.add_argument(
        "--output-csv",
        default="data/datasets/Actions/action_har_labels.csv",
    )
    hf_parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
    )
    hf_parser.add_argument(
        "--no-check-images",
        action="store_true",
    )
    hf_parser.set_defaults(func=command_from_huggingface)

    return parser

def map_hf_label_to_action(label_name: str) -> List[str]:
    label_name = label_name.lower().strip()

    mapping = {
        "running": ["running", "moving", "fast_motion"],
        "walking": ["walking", "moving"],
        "cycling": ["cycling", "moving"],
        "dancing": ["dancing", "moving"],
        "sitting": ["sitting", "still"],
        "sleeping": ["lying_down", "still"],
        "drinking": ["drinking"],
        "eating": ["eating"],

        "calling": ["standing", "still"],
        "clapping": ["arms_raised", "moving"],
        "fighting": ["moving", "fast_motion"],
        "hugging": ["standing", "still"],
        "laughing": ["standing", "still"],
        "listening_to_music": ["sitting", "still"],
        "texting": ["sitting", "still"],
        "using_laptop": ["working", "sitting", "still"],
    }

    return mapping.get(label_name, [])

def command_from_huggingface(args) -> None:
    dataset_name = args.dataset_name
    output_dir = Path(args.output_image_dir)
    output_csv = Path(args.output_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name)

    split_name = args.split

    if split_name not in dataset:
        split_name = list(dataset.keys())[0]

    data = dataset[split_name]

    rows = []

    label_names = data.features["labels"].names if "labels" in data.features else None

    total = len(data)

    if args.max_rows is not None:
        total = min(total, args.max_rows)

    for index in range(total):
        item = data[index]

        image = item["image"]

        label_id = item["labels"]

        if label_names is not None:
            label_name = label_names[label_id]
        else:
            label_name = str(label_id)

        action_labels = map_hf_label_to_action(label_name)

        if not action_labels:
            continue

        image_path = output_dir / f"{index:06d}_{label_name}.jpg"
        image.convert("RGB").save(image_path, quality=95)

        row = {
            "frame_path": str(image_path),
            "source_dataset": dataset_name,
            "source_label": label_name,
        }

        for label in LABELS:
            row[label] = 0

        for label in action_labels:
            if label in LABELS:
                row[label] = 1

        rows.append(row)

    df = pd.DataFrame(rows)

    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    df.to_csv(output_csv, index=False)

    print("\nDataset Actions criado via Hugging Face:")
    print(output_csv)
    print("Dataset:", dataset_name)
    print("Split:", split_name)
    print("Linhas:", len(df))

    print_dataset_summary(df)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()