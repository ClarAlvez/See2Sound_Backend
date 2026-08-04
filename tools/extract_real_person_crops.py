import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from ultralytics import YOLO


LABEL_COLUMNS = [
    "person",
    "man",
    "woman",

    "short_hair",
    "long_hair",
    "bald_hair",
    "black_hair",
    "blonde_hair",
    "brown_hair",
    "gray_hair",
    "straight_hair",
    "wavy_hair",
    "bangs_hair",
    "receding_hairline",

    "black_clothes",
    "white_clothes",
    "red_clothes",
    "blue_clothes",
    "green_clothes",
    "yellow_clothes",
    "brown_clothes",
    "gray_clothes",
    "orange_clothes",
    "pink_clothes",
    "purple_clothes",

    "dress",
    "glasses",
    "hat",
    "backpack",
    "bag",
]


def crop_people_from_image(
    model: YOLO,
    image_path: Path,
    output_dir: Path,
    confidence_threshold: float,
    padding_ratio: float,
    max_people: int,
) -> List[Dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    results = model(str(image_path), verbose=False)

    if not results or results[0].boxes is None:
        return []

    detected_people = []

    for index, box in enumerate(results[0].boxes):
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        # COCO class 0 = person
        if class_id != 0:
            continue

        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = box_width * padding_ratio
        pad_y = box_height * padding_ratio

        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(image_width, int(x2 + pad_x))
        crop_y2 = min(image_height, int(y2 + pad_y))

        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        crop_filename = f"{image_path.stem}_person_{index:02d}.jpg"
        crop_path = output_dir / crop_filename
        crop.save(crop_path, quality=95)

        detected_people.append(
            {
                "frame_path": str(crop_path),
                "source_dataset": "real_pipeline_crop",
                "source_frame_path": str(image_path),
                "detector_confidence": round(confidence, 4),
                "bbox_x1": crop_x1,
                "bbox_y1": crop_y1,
                "bbox_x2": crop_x2,
                "bbox_y2": crop_y2,
            }
        )

    detected_people = sorted(
        detected_people,
        key=lambda item: item["detector_confidence"],
        reverse=True,
    )

    return detected_people[:max_people]


def collect_frame_paths(input_dir: Path) -> List[Path]:
    image_paths = (
        list(input_dir.rglob("*.jpg"))
        + list(input_dir.rglob("*.jpeg"))
        + list(input_dir.rglob("*.png"))
    )

    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Extrai crops reais de pessoas para revisão manual da SpectraPersonNet."
    )

    parser.add_argument(
        "--input-frames-dir",
        required=True,
        help="Pasta contendo frames reais extraídos do pipeline.",
    )

    parser.add_argument(
        "--output-crops-dir",
        default="data/datasets/manual_review/person_crops",
        help="Pasta onde os crops serão salvos.",
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/manual_review/spectra_person_real_crops_review.csv",
        help="CSV inicial para revisão manual.",
    )

    parser.add_argument(
        "--model-name",
        default="yolov8n.pt",
        help="Modelo YOLO usado para detectar pessoas.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--max-people-per-frame",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    input_frames_dir = Path(args.input_frames_dir)
    output_crops_dir = Path(args.output_crops_dir)
    output_csv = Path(args.output_csv)

    if not input_frames_dir.exists():
        raise FileNotFoundError(f"Pasta de frames não encontrada: {input_frames_dir}")

    output_crops_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = collect_frame_paths(input_frames_dir)

    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    print("Frames encontrados:", len(frame_paths))
    print("Carregando YOLO:", args.model_name)

    model = YOLO(args.model_name)

    rows = []

    for index, frame_path in enumerate(frame_paths):
        crops = crop_people_from_image(
            model=model,
            image_path=frame_path,
            output_dir=output_crops_dir,
            confidence_threshold=args.confidence_threshold,
            padding_ratio=args.padding_ratio,
            max_people=args.max_people_per_frame,
        )

        for crop in crops:
            row = dict(crop)

            # Labels iniciais para revisão manual.
            # person começa como 1 porque o crop veio do detector de pessoa.
            for label in LABEL_COLUMNS:
                row[label] = 0

            row["person"] = 1

            rows.append(row)

        if (index + 1) % 50 == 0:
            print(f"Frames processados: {index + 1}/{len(frame_paths)} | crops: {len(rows)}")

    fieldnames = [
        "frame_path",
        "source_dataset",
        "source_frame_path",
        "detector_confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ] + LABEL_COLUMNS

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nCrops extraídos com sucesso.")
    print("Total de crops:", len(rows))
    print("Pasta de crops:", output_crops_dir)
    print("CSV para revisão:", output_csv)


if __name__ == "__main__":
    main()