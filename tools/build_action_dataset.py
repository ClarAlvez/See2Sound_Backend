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
        "calling": ["standing", "still"],
        "clapping": ["arms_raised", "moving"],
        "fighting": ["moving", "fast_motion"],
        "hugging": ["standing", "still"],
        "laughing": ["standing", "still"],
        "listening_to_music": ["sitting", "still"],
        "texting": ["working", "sitting", "still"],
        "using_laptop": ["working", "sitting", "still"],

        # Stanford40
        "applauding": ["arms_raised", "moving"],
        "blowing_bubbles": ["standing", "still"],
        "brushing_teeth": ["standing", "still"],
        "cleaning_the_floor": ["working", "moving"],
        "climbing": ["exercising", "moving"],
        "cooking": ["working", "standing", "still"],
        "cutting_trees": ["working", "moving"],
        "cutting_vegetables": ["working", "standing", "still"],
        "feeding_a_horse": ["standing", "still"],
        "fishing": ["standing", "still"],
        "fixing_a_bike": ["working", "crouching", "still"],
        "fixing_a_car": ["working", "crouching", "still"],
        "gardening": ["working", "crouching", "moving"],
        "holding_an_umbrella": ["standing", "still"],
        "jumping": ["jumping", "moving", "fast_motion"],
        "looking_through_a_microscope": ["working", "sitting", "still"],
        "looking_through_a_telescope": ["standing", "still"],
        "playing_guitar": ["playing", "sitting", "still"],
        "playing_violin": ["playing", "standing", "still"],
        "pouring_liquid": ["drinking", "standing", "still"],
        "pushing_a_cart": ["walking", "moving"],
        "reading": ["sitting", "still"],
        "phoning": ["standing", "still"],
        "riding_a_bike": ["cycling", "moving"],
        "riding_a_horse": ["moving"],
        "rowing_a_boat": ["exercising", "moving"],
        "shooting_an_arrow": ["arms_raised", "standing", "still"],
        "smoking": ["standing", "still"],
        "taking_photos": ["standing", "still"],
        "texting_message": ["working", "sitting", "still"],
        "throwing_frisby": ["arms_raised", "moving"],
        "using_a_computer": ["working", "sitting", "still"],
        "walking_the_dog": ["walking", "moving"],
        "washing_dishes": ["working", "standing", "still"],
        "watching_tv": ["sitting", "still"],
        "waving_hands": ["arms_raised", "moving"],
        "writing_on_a_board": ["working", "standing", "still"],
        "writing_on_a_book": ["working", "sitting", "still"],

        # UCF101
        "apply_eye_makeup": ["standing", "still"],
        "apply_lipstick": ["standing", "still"],
        "archery": ["arms_raised", "standing", "still"],
        "baby_crawling": ["crouching", "moving"],
        "balance_beam": ["walking", "moving"],
        "band_marching": ["walking", "playing", "moving"],
        "baseball_pitch": ["arms_raised", "moving", "fast_motion"],
        "basketball": ["playing", "running", "moving"],
        "basketball_dunk": ["jumping", "playing", "moving", "fast_motion"],
        "bench_press": ["exercising", "lying_down", "moving"],
        "biking": ["cycling", "moving"],
        "billiards": ["playing", "standing", "still"],
        "blow_dry_hair": ["standing", "still"],
        "blowing_candles": ["standing", "still"],
        "body_weight_squats": ["exercising", "crouching", "moving"],
        "bowling": ["playing", "moving"],
        "boxing_punching_bag": ["exercising", "moving", "fast_motion"],
        "boxing_speed_bag": ["exercising", "moving", "fast_motion"],
        "breast_stroke": ["exercising", "moving"],
        "brushing_teeth": ["standing", "still"],
        "clean_and_jerk": ["exercising", "arms_raised", "moving"],
        "cliff_diving": ["jumping", "moving", "fast_motion"],
        "cricket_bowling": ["arms_raised", "moving", "fast_motion"],
        "cricket_shot": ["playing", "moving"],
        "cutting_in_kitchen": ["working", "standing", "still"],
        "diving": ["jumping", "moving", "fast_motion"],
        "drumming": ["playing", "sitting", "moving"],
        "fencing": ["exercising", "moving"],
        "field_hockey_penalty": ["playing", "moving"],
        "floor_gymnastics": ["exercising", "moving", "fast_motion"],
        "frisbee_catch": ["arms_raised", "playing", "moving"],
        "front_crawl": ["exercising", "moving"],
        "golf_swing": ["playing", "moving"],
        "haircut": ["standing", "still"],
        "hammering": ["working", "moving"],
        "hammer_throw": ["arms_raised", "exercising", "moving", "fast_motion"],
        "handstand_pushups": ["exercising", "arms_raised", "moving"],
        "handstand_walking": ["exercising", "moving"],
        "head_massage": ["standing", "still"],
        "high_jump": ["jumping", "exercising", "moving", "fast_motion"],
        "horse_race": ["moving", "fast_motion"],
        "horse_riding": ["moving"],
        "hula_hoop": ["dancing", "moving"],
        "ice_dancing": ["dancing", "moving"],
        "javelin_throw": ["arms_raised", "moving", "fast_motion"],
        "juggling_balls": ["playing", "standing", "moving"],
        "jump_rope": ["jumping", "exercising", "moving", "fast_motion"],
        "jumping_jack": ["jumping", "exercising", "moving", "fast_motion"],
        "kayaking": ["exercising", "moving"],
        "knitting": ["sitting", "still"],
        "long_jump": ["jumping", "running", "moving", "fast_motion"],
        "lunges": ["exercising", "walking", "moving"],
        "military_parade": ["walking", "moving"],
        "mixing": ["working", "standing", "still"],
        "mopping_floor": ["working", "moving"],
        "nunchucks": ["exercising", "moving", "fast_motion"],
        "parallel_bars": ["exercising", "moving"],
        "pizza_tossing": ["working", "standing", "moving"],
        "playing_cello": ["playing", "sitting", "still"],
        "playing_daf": ["playing", "sitting", "moving"],
        "playing_dhol": ["playing", "standing", "moving"],
        "playing_flute": ["playing", "standing", "still"],
        "playing_guitar": ["playing", "sitting", "still"],
        "playing_piano": ["playing", "sitting", "still"],
        "playing_sitar": ["playing", "sitting", "still"],
        "playing_tabla": ["playing", "sitting", "moving"],
        "playing_violin": ["playing", "standing", "still"],
        "pole_vault": ["jumping", "exercising", "moving", "fast_motion"],
        "pommel_horse": ["exercising", "moving"],
        "pull_ups": ["exercising", "moving"],
        "punch": ["exercising", "moving", "fast_motion"],
        "push_ups": ["exercising", "lying_down", "moving"],
        "rafting": ["exercising", "moving"],
        "rock_climbing_indoor": ["exercising", "moving"],
        "rope_climbing": ["exercising", "moving"],
        "rowing": ["exercising", "moving"],
        "salsa_spin": ["dancing", "moving"],
        "shaving_beard": ["standing", "still"],
        "shotput": ["arms_raised", "exercising", "moving"],
        "skate_boarding": ["moving"],
        "skiing": ["moving", "fast_motion"],
        "skijet": ["moving", "fast_motion"],
        "sky_diving": ["falling", "moving", "fast_motion"],
        "soccer_juggling": ["playing", "moving"],
        "soccer_penalty": ["playing", "running", "moving"],
        "still_rings": ["exercising", "moving"],
        "sumo_wrestling": ["exercising", "moving"],
        "surfing": ["moving"],
        "swing": ["playing", "moving"],
        "table_tennis_shot": ["playing", "moving"],
        "tai_chi": ["exercising", "standing", "moving"],
        "tennis_swing": ["playing", "moving"],
        "throw_discus": ["arms_raised", "moving", "fast_motion"],
        "trampoline_jumping": ["jumping", "moving", "fast_motion"],
        "typing": ["working", "sitting", "still"],
        "uneven_bars": ["exercising", "moving"],
        "volleyball_spiking": ["jumping", "playing", "moving"],
        "walking_with_dog": ["walking", "moving"],
        "wall_pushups": ["exercising", "standing", "moving"],
        "writing_on_board": ["working", "standing", "still"],
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

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()