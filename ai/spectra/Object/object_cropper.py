from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ai.spectra.Object.labels import LABELS


YOLO_TO_SPECTRA_OBJECT = {
    # COCO / YOLO labels
    "chair": ["chair"],
    "couch": ["sofa"],
    "sofa": ["sofa"],
    "bed": ["bed"],
    "dining table": ["table"],
    "table": ["table"],
    "toilet": ["toilet"],
    "bench": ["chair"],

    "cell phone": ["phone", "screen"],
    "mobile phone": ["phone", "screen"],
    "laptop": ["computer", "screen"],
    "tv": ["television", "screen"],
    "television": ["television", "screen"],
    "keyboard": ["keyboard", "computer"],
    "mouse": ["mouse", "computer"],
    "remote": ["remote"],

    "car": ["car"],
    "bicycle": ["bicycle"],
    "motorcycle": ["motorcycle"],
    "bus": ["bus"],
    "truck": ["truck"],
    "train": ["train"],
    "boat": ["boat"],
    "airplane": ["airplane"],

    "dog": ["dog", "animal"],
    "cat": ["cat", "animal"],
    "bird": ["bird", "animal"],
    "horse": ["horse", "animal"],
    "sheep": ["sheep", "animal"],
    "cow": ["cow", "animal"],
    "elephant": ["animal"],
    "bear": ["animal"],
    "zebra": ["animal"],
    "giraffe": ["animal"],

    "banana": ["fruit", "food"],
    "apple": ["fruit", "food"],
    "orange": ["fruit", "food"],
    "broccoli": ["food"],
    "carrot": ["food"],
    "hot dog": ["food"],
    "pizza": ["food"],
    "donut": ["food"],
    "cake": ["food"],
    "sandwich": ["food"],

    "cup": ["cup"],
    "bottle": ["bottle"],
    "wine glass": ["cup"],
    "bowl": ["bowl"],
    "plate": ["plate"],
    "fork": ["fork"],
    "spoon": ["spoon"],
    "knife": ["knife"],

    "backpack": ["bag", "backpack"],
    "handbag": ["bag", "handbag"],
    "suitcase": ["bag", "suitcase"],
    "umbrella": ["umbrella"],

    "book": ["book", "document"],

    "sports ball": ["ball", "toy"],
    "frisbee": ["toy"],
    "kite": ["kite", "toy"],
    "skateboard": ["skateboard", "toy"],
    "surfboard": ["surfboard"],
    "tennis racket": ["sports_racket"],
    "baseball bat": ["sports_racket"],
    "baseball glove": ["sports_racket"],

    "teddy bear": ["toy"],
    "scissors": ["knife"],

    # Ignorar pessoa aqui, porque Person já cuida disso
    "person": [],
}


def map_yolo_label_to_spectra(label_name: str) -> List[str]:
    label_name = str(label_name).strip().lower()

    mapped_labels = YOLO_TO_SPECTRA_OBJECT.get(label_name, [])

    if mapped_labels is None:
        return []

    if isinstance(mapped_labels, str):
        mapped_labels = [mapped_labels]

    return sorted({
        label
        for label in mapped_labels
        if label in LABELS
    })


def expand_box(
    box: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.08,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    new_x1 = max(0, int(x1 - pad_x))
    new_y1 = max(0, int(y1 - pad_y))
    new_x2 = min(image_width, int(x2 + pad_x))
    new_y2 = min(image_height, int(y2 + pad_y))

    return new_x1, new_y1, new_x2, new_y2


class ObjectCropper:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        max_objects: int = 20,
        min_area_ratio: float = 0.002,
        padding_ratio: float = 0.08,
        device: Optional[str] = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "Ultralytics não está instalado. Instale com: pip install ultralytics"
            ) from error

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_objects = max_objects
        self.min_area_ratio = min_area_ratio
        self.padding_ratio = padding_ratio
        self.device = device

        self.model = YOLO(model_name)

    def crop_objects(
        self,
        image_path: str,
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        image_area = image_width * image_height

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        names = result.names
        crop_results = []

        boxes = result.boxes

        for index, box in enumerate(boxes):
            if len(crop_results) >= self.max_objects:
                break

            class_id = int(box.cls.item())
            detector_label = names.get(class_id, str(class_id))
            detector_score = float(box.conf.item())

            mapped_labels = map_yolo_label_to_spectra(detector_label)

            # Ignora classes que não fazem parte do Object da Spectra
            if not mapped_labels:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bbox_width = max(0.0, x2 - x1)
            bbox_height = max(0.0, y2 - y1)
            bbox_area = bbox_width * bbox_height

            if image_area > 0 and (bbox_area / image_area) < self.min_area_ratio:
                continue

            crop_box = expand_box(
                box=(x1, y1, x2, y2),
                image_width=image_width,
                image_height=image_height,
                padding_ratio=self.padding_ratio,
            )

            crop = image.crop(crop_box)

            stem = image_path.stem
            crop_filename = (
                f"{stem}_object_{len(crop_results):03d}_{detector_label.replace(' ', '_')}.jpg"
            )
            crop_path = output_dir / crop_filename
            crop.save(crop_path, quality=95)

            crop_results.append(
                {
                    "crop_path": str(crop_path),
                    "source_image_path": str(image_path),
                    "detector_label": detector_label,
                    "detector_score": round(detector_score, 4),
                    "mapped_labels": mapped_labels,
                    "bbox_xyxy": {
                        "x1": crop_box[0],
                        "y1": crop_box[1],
                        "x2": crop_box[2],
                        "y2": crop_box[3],
                    },
                }
            )

        return crop_results