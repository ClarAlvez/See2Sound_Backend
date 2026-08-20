import argparse
from pathlib import Path
from typing import List
from datasets import load_dataset

import cv2
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

    for raw_label in raw_labels:
        normalized_label = normalize_action_name(raw_label)

        if normalized_label in LABELS:
            valid_labels.append(normalized_label)
            continue

        mapped_labels = map_hf_label_to_action(normalized_label)

        for mapped_label in mapped_labels:
            if mapped_label in LABELS:
                valid_labels.append(mapped_label)

    return sorted(set(valid_labels))


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

    statefarm_parser = subparsers.add_parser("from-statefarm")
    statefarm_parser.add_argument(
        "--input-dir",
        default="data/external/actions/state_farm/train",
    )
    statefarm_parser.add_argument(
        "--output-csv",
        default="data/datasets/Actions/action_statefarm_labels.csv",
    )
    statefarm_parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
    )
    statefarm_parser.add_argument(
        "--no-check-images",
        action="store_true",
    )
    statefarm_parser.set_defaults(func=command_from_statefarm)

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

    ucf_parser = subparsers.add_parser("from-ucf101")
    ucf_parser.add_argument(
        "--input-dir",
        default="data/external/actions/UCF101",
    )
    ucf_parser.add_argument(
        "--output-image-dir",
        default="data/external/actions/ucf101_frames",
    )
    ucf_parser.add_argument(
        "--output-csv",
        default="data/datasets/Actions/action_ucf101_labels.csv",
    )
    ucf_parser.add_argument(
        "--frames-per-video",
        type=int,
        default=3,
    )
    ucf_parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
    )
    ucf_parser.add_argument(
        "--no-check-images",
        action="store_true",
    )
    ucf_parser.set_defaults(func=command_from_ucf101)

    return parser

def map_hf_label_to_action(label_name: str) -> List[str]:
    label_name = normalize_action_name(label_name)

    mapping = {
        # Human Action Recognition
        "running": ["running", "moving", "fast_motion"],
        "walking": ["walking", "moving"],
        "cycling": ["cycling", "moving"],
        "dancing": ["dancing", "moving"],
        "sitting": ["sitting", "still"],
        "sleeping": ["lying_down", "still"],
        "drinking": ["drinking", "still"],
        "eating": ["eating", "still"],
        "calling": ["phone_use", "talking", "standing", "still"],
        "clapping": ["arms_raised", "moving"],
        "fighting": ["martial_activity", "sports", "moving", "fast_motion"],
        "hugging": ["standing", "still"],
        "laughing": ["talking", "standing", "still"],
        "listening_to_music": ["sitting", "still"],
        "texting": ["phone_use", "working", "sitting", "still"],
        "using_laptop": ["computer_use", "working", "sitting", "still"],

        # Stanford40
        "applauding": ["arms_raised", "moving"],
        "blowing_bubbles": ["standing", "still"],
        "brushing_teeth": ["grooming", "standing", "still"],
        "cleaning_the_floor": ["cleaning", "working", "moving"],
        "climbing": ["climbing", "exercising", "moving"],
        "cooking": ["cooking", "working", "standing", "still"],
        "cutting_trees": ["working", "moving"],
        "cutting_vegetables": ["cooking", "working", "standing", "still"],
        "feeding_a_horse": ["standing", "still"],
        "fishing": ["sports", "standing", "still"],
        "fixing_a_bike": ["working", "crouching", "still"],
        "fixing_a_car": ["working", "crouching", "still"],
        "gardening": ["working", "crouching", "moving"],
        "holding_an_umbrella": ["standing", "still"],
        "jumping": ["jumping", "moving", "fast_motion"],
        "looking_through_a_microscope": ["working", "sitting", "still"],
        "looking_through_a_telescope": ["standing", "still"],
        "playing_guitar": ["instrument_playing", "playing", "sitting", "still"],
        "playing_violin": ["instrument_playing", "playing", "standing", "still"],
        "pouring_liquid": ["drinking", "standing", "still"],
        "pushing_a_cart": ["carrying", "walking", "moving"],
        "reading": ["reading", "sitting", "still"],
        "phoning": ["phone_use", "talking", "standing", "still"],
        "riding_a_bike": ["cycling", "sports", "moving"],
        "riding_a_horse": ["sports", "moving"],
        "rowing_a_boat": ["water_activity", "sports", "exercising", "moving"],
        "shooting_an_arrow": ["sports", "arms_raised", "standing", "still"],
        "smoking": ["standing", "still"],
        "taking_photos": ["standing", "still"],
        "texting_message": ["phone_use", "working", "sitting", "still"],
        "throwing_frisby": ["throwing", "sports", "arms_raised", "moving"],
        "using_a_computer": ["computer_use", "working", "sitting", "still"],
        "walking_the_dog": ["walking", "moving"],
        "washing_dishes": ["cleaning", "working", "standing", "still"],
        "watching_tv": ["sitting", "still"],
        "waving_hands": ["arms_raised", "moving"],
        "writing_on_a_board": ["writing", "working", "standing", "still"],
        "writing_on_a_book": ["writing", "working", "sitting", "still"],

        # UCF101
        "apply_eye_makeup": ["makeup", "grooming", "standing", "still"],
        "apply_lipstick": ["makeup", "grooming", "standing", "still"],
        "archery": ["sports", "arms_raised", "standing", "still"],
        "baby_crawling": ["crouching", "moving"],
        "balance_beam": ["sports", "walking", "moving"],
        "band_marching": ["instrument_playing", "walking", "playing", "moving"],
        "baseball_pitch": ["sports", "ball_sport", "throwing", "arms_raised", "moving", "fast_motion"],
        "basketball": ["sports", "ball_sport", "playing", "running", "moving"],
        "basketball_dunk": ["sports", "ball_sport", "jumping", "playing", "moving", "fast_motion"],
        "bench_press": ["sports", "exercising", "lying_down", "moving"],
        "biking": ["cycling", "sports", "moving"],
        "billiards": ["sports", "ball_sport", "playing", "standing", "still"],
        "blow_dry_hair": ["grooming", "standing", "still"],
        "blowing_candles": ["standing", "still"],
        "body_weight_squats": ["sports", "exercising", "crouching", "moving"],
        "bowling": ["sports", "ball_sport", "playing", "throwing", "moving"],
        "boxing_punching_bag": ["sports", "martial_activity", "exercising", "moving", "fast_motion"],
        "boxing_speed_bag": ["sports", "martial_activity", "exercising", "moving", "fast_motion"],
        "breast_stroke": ["swimming", "water_activity", "sports", "exercising", "moving"],
        "brushing_teeth": ["grooming", "standing", "still"],
        "clean_and_jerk": ["sports", "exercising", "arms_raised", "moving"],
        "cliff_diving": ["water_activity", "sports", "jumping", "falling", "moving", "fast_motion"],
        "cricket_bowling": ["sports", "ball_sport", "throwing", "arms_raised", "moving", "fast_motion"],
        "cricket_shot": ["sports", "ball_sport", "playing", "moving"],
        "cutting_in_kitchen": ["cooking", "working", "standing", "still"],
        "diving": ["water_activity", "sports", "jumping", "falling", "moving", "fast_motion"],
        "drumming": ["instrument_playing", "playing", "sitting", "moving"],
        "fencing": ["sports", "martial_activity", "exercising", "moving"],
        "field_hockey_penalty": ["sports", "ball_sport", "playing", "moving"],
        "floor_gymnastics": ["sports", "exercising", "moving", "fast_motion"],
        "frisbee_catch": ["sports", "ball_sport", "throwing", "arms_raised", "playing", "moving"],
        "front_crawl": ["swimming", "water_activity", "sports", "exercising", "moving"],
        "golf_swing": ["sports", "ball_sport", "playing", "moving"],
        "haircut": ["grooming", "standing", "still"],
        "hammering": ["working", "moving"],
        "hammer_throw": ["sports", "throwing", "arms_raised", "exercising", "moving", "fast_motion"],
        "handstand_pushups": ["sports", "exercising", "arms_raised", "moving"],
        "handstand_walking": ["sports", "exercising", "moving"],
        "head_massage": ["grooming", "standing", "still"],
        "high_jump": ["sports", "jumping", "exercising", "moving", "fast_motion"],
        "horse_race": ["sports", "moving", "fast_motion"],
        "horse_riding": ["sports", "moving"],
        "hula_hoop": ["dancing", "exercising", "moving"],
        "ice_dancing": ["dancing", "sports", "moving"],
        "javelin_throw": ["sports", "throwing", "arms_raised", "moving", "fast_motion"],
        "juggling_balls": ["sports", "ball_sport", "playing", "standing", "moving"],
        "jump_rope": ["sports", "jumping", "exercising", "moving", "fast_motion"],
        "jumping_jack": ["sports", "jumping", "exercising", "moving", "fast_motion"],
        "kayaking": ["water_activity", "sports", "exercising", "moving"],
        "knitting": ["working", "sitting", "still"],
        "long_jump": ["sports", "jumping", "running", "moving", "fast_motion"],
        "lunges": ["sports", "exercising", "walking", "moving"],
        "military_parade": ["walking", "moving"],
        "mixing": ["cooking", "working", "standing", "still"],
        "mopping_floor": ["cleaning", "working", "moving"],
        "nunchucks": ["sports", "martial_activity", "exercising", "moving", "fast_motion"],
        "parallel_bars": ["sports", "exercising", "moving"],
        "pizza_tossing": ["cooking", "working", "standing", "moving"],
        "playing_cello": ["instrument_playing", "playing", "sitting", "still"],
        "playing_daf": ["instrument_playing", "playing", "sitting", "moving"],
        "playing_dhol": ["instrument_playing", "playing", "standing", "moving"],
        "playing_flute": ["instrument_playing", "playing", "standing", "still"],
        "playing_guitar": ["instrument_playing", "playing", "sitting", "still"],
        "playing_piano": ["instrument_playing", "playing", "sitting", "still"],
        "playing_sitar": ["instrument_playing", "playing", "sitting", "still"],
        "playing_tabla": ["instrument_playing", "playing", "sitting", "moving"],
        "playing_violin": ["instrument_playing", "playing", "standing", "still"],
        "pole_vault": ["sports", "jumping", "exercising", "moving", "fast_motion"],
        "pommel_horse": ["sports", "exercising", "moving"],
        "pull_ups": ["sports", "exercising", "moving"],
        "punch": ["sports", "martial_activity", "exercising", "moving", "fast_motion"],
        "push_ups": ["sports", "exercising", "lying_down", "moving"],
        "rafting": ["water_activity", "sports", "exercising", "moving"],
        "rock_climbing_indoor": ["climbing", "sports", "exercising", "moving"],
        "rope_climbing": ["climbing", "sports", "exercising", "moving"],
        "rowing": ["water_activity", "sports", "exercising", "moving"],
        "salsa_spin": ["dancing", "moving"],
        "shaving_beard": ["grooming", "standing", "still"],
        "shotput": ["sports", "throwing", "arms_raised", "exercising", "moving"],
        "skate_boarding": ["sports", "moving"],
        "skiing": ["sports", "moving", "fast_motion"],
        "skijet": ["water_activity", "sports", "moving", "fast_motion"],
        "sky_diving": ["sports", "falling", "moving", "fast_motion"],
        "soccer_juggling": ["sports", "ball_sport", "playing", "moving"],
        "soccer_penalty": ["sports", "ball_sport", "playing", "running", "moving"],
        "still_rings": ["sports", "exercising", "moving"],
        "sumo_wrestling": ["sports", "martial_activity", "exercising", "moving"],
        "surfing": ["water_activity", "sports", "moving"],
        "swing": ["playing", "moving"],
        "table_tennis_shot": ["sports", "racket_sport", "ball_sport", "playing", "moving"],
        "tai_chi": ["sports", "martial_activity", "exercising", "standing", "moving"],
        "tennis_swing": ["sports", "racket_sport", "ball_sport", "playing", "moving"],
        "throw_discus": ["sports", "throwing", "arms_raised", "moving", "fast_motion"],
        "trampoline_jumping": ["sports", "jumping", "moving", "fast_motion"],
        "typing": ["computer_use", "working", "sitting", "still"],
        "uneven_bars": ["sports", "exercising", "moving"],
        "volleyball_spiking": ["sports", "ball_sport", "jumping", "playing", "moving"],
        "walking_with_dog": ["walking", "moving"],
        "wall_pushups": ["sports", "exercising", "standing", "moving"],
        "writing_on_board": ["writing", "working", "standing", "still"],
        "yo_yo": ["playing", "standing", "still"],
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

VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov", ".mkv"]


def normalize_action_name(value: str) -> str:
    value = str(value).strip()

    normalized = ""

    for index, char in enumerate(value):
        if char.isupper() and index > 0 and value[index - 1].islower():
            normalized += "_"

        normalized += char

    return (
        normalized
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def collect_videos(input_dir: Path) -> List[Path]:
    video_paths = []

    for extension in VIDEO_EXTENSIONS:
        video_paths.extend(input_dir.rglob(f"*{extension}"))

    return sorted(video_paths)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frames_per_video: int = 3,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"Não foi possível abrir vídeo: {video_path}")
        return []

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        capture.release()
        return []

    if frames_per_video <= 1:
        selected_indices = [frame_count // 2]
    else:
        selected_indices = [
            int((i + 1) * frame_count / (frames_per_video + 1))
            for i in range(frames_per_video)
        ]

    saved_paths = []

    for frame_index in selected_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = capture.read()

        if not success or frame is None:
            continue

        output_path = output_dir / f"{video_path.stem}_frame_{frame_index:06d}.jpg"

        cv2.imwrite(str(output_path), frame)
        saved_paths.append(output_path)

    capture.release()

    return saved_paths


def infer_ucf101_label_from_path(video_path: Path, input_dir: Path) -> str:
    relative_path = video_path.relative_to(input_dir)

    if len(relative_path.parts) < 2:
        return ""

    return normalize_action_name(relative_path.parts[0])


def command_from_ucf101(args) -> None:
    input_dir = Path(args.input_dir)
    output_image_dir = Path(args.output_image_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta UCF101 não encontrada: {input_dir}")

    videos = collect_videos(input_dir)

    if args.max_videos is not None:
        videos = videos[:args.max_videos]

    rows = []
    skipped = 0

    for video_path in videos:
        source_label = infer_ucf101_label_from_path(
            video_path=video_path,
            input_dir=input_dir,
        )

        action_labels = map_hf_label_to_action(source_label)

        if not action_labels:
            skipped += 1
            continue

        video_output_dir = output_image_dir / source_label

        frame_paths = extract_frames_from_video(
            video_path=video_path,
            output_dir=video_output_dir,
            frames_per_video=args.frames_per_video,
        )

        for frame_path in frame_paths:
            row = {
                "frame_path": str(frame_path),
                "source_dataset": "ucf101",
                "source_label": source_label,
                "source_video_path": str(video_path),
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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nDataset Actions criado a partir do UCF101:")
    print(output_csv)
    print("Vídeos encontrados:", len(videos))
    print("Frames salvos:", len(df))
    print("Vídeos/classes ignorados:", skipped)

    print_dataset_summary(df)

def map_statefarm_label_to_action(label_name: str) -> List[str]:
    label_name = normalize_action_name(label_name)

    mapping = {
        "c0": ["driving", "sitting", "still"],
        "c1": ["driving", "phone_use", "sitting"],
        "c2": ["driving", "phone_use", "talking", "sitting"],
        "c3": ["driving", "phone_use", "sitting"],
        "c4": ["driving", "phone_use", "talking", "sitting"],
        "c5": ["driving", "radio_use", "working", "sitting"],
        "c6": ["driving", "drinking", "sitting"],
        "c7": ["driving", "reaching", "arms_raised", "sitting"],
        "c8": ["driving", "makeup", "grooming", "sitting"],
        "c9": ["driving", "talking", "sitting"],
    }

    return mapping.get(label_name, [])


def command_from_statefarm(args) -> None:
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta State Farm não encontrada: {input_dir}")

    image_paths = collect_images(input_dir)

    rows = []
    skipped = 0

    if args.max_rows is not None:
        image_paths = image_paths[:args.max_rows]

    for image_path in image_paths:
        relative_path = image_path.relative_to(input_dir)

        if len(relative_path.parts) < 2:
            skipped += 1
            continue

        source_label = normalize_action_name(relative_path.parts[0])
        action_labels = map_statefarm_label_to_action(source_label)

        if not action_labels:
            skipped += 1
            continue

        row = {
            "frame_path": str(image_path),
            "source_dataset": "state_farm_distracted_driver",
            "source_label": source_label,
        }

        for label in LABELS:
            row[label] = 0

        for label in action_labels:
            if label in LABELS:
                row[label] = 1

        rows.append(row)

    if not rows:
        raise ValueError(
            "Dataset State Farm vazio. Confira se a pasta possui train/c0, train/c1, ..., train/c9."
        )

    df = pd.DataFrame(rows)

    df = clean_dataset(
        df,
        check_images=not args.no_check_images,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nDataset Actions criado a partir do State Farm:")
    print(output_csv)
    print("Imagens encontradas:", len(image_paths))
    print("Linhas salvas:", len(df))
    print("Ignoradas:", skipped)

    print_dataset_summary(df)

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()