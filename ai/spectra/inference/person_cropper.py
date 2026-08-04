from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


class PersonCropper:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        padding_ratio: float = 0.08,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.padding_ratio = padding_ratio

    def crop_people(
        self,
        image_path: str,
        output_dir: str,
        max_people: int = 3,
    ) -> List[Dict[str, Any]]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        results = self.model(str(image_path), verbose=False)

        crops = []

        if not results:
            return crops

        result = results[0]

        if result.boxes is None:
            return crops

        boxes = result.boxes

        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            # COCO class 0 = person
            if class_id != 0:
                continue

            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = x2 - x1
            box_height = y2 - y1

            pad_x = box_width * self.padding_ratio
            pad_y = box_height * self.padding_ratio

            crop_x1 = max(0, int(x1 - pad_x))
            crop_y1 = max(0, int(y1 - pad_y))
            crop_x2 = min(image_width, int(x2 + pad_x))
            crop_y2 = min(image_height, int(y2 + pad_y))

            crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            crop_path = output_dir / f"{image_path.stem}_person_{index:02d}.jpg"
            crop.save(crop_path, quality=95)

            crops.append(
                {
                    "crop_path": str(crop_path),
                    "bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
                    "detector_confidence": confidence,
                }
            )

        crops = sorted(
            crops,
            key=lambda item: item["detector_confidence"],
            reverse=True,
        )

        return crops[:max_people]