from typing import Any, Dict, List


class CorrelationObjectAdapter:
    GENERIC_LABELS = {
        "animal",
        "food",
        "fruit",
        "bag",
        "screen",
        "document",
        "toy",
    }

    def process_result(
        self,
        object_result: Dict[str, Any],
        frame_id: int,
        timestamp: float,
        scene_id=None,
    ) -> List[Dict[str, Any]]:
        if not object_result:
            return []

        raw_objects = (
            object_result.get(
                "crop_results"
            )
            or object_result.get(
                "raw_crop_results"
            )
            or object_result.get(
                "objects"
            )
            or []
        )

        objects = []

        for index, item in enumerate(
            raw_objects
        ):
            normalized = (
                self._normalize_item(
                    item=item,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    scene_id=scene_id,
                    index=index,
                )
            )

            if normalized is not None:
                objects.append(
                    normalized
                )

        return objects

    def _normalize_item(
        self,
        item,
        frame_id,
        timestamp,
        scene_id,
        index,
    ):
        crop = item.get(
            "crop",
            item,
        )

        result = item.get(
            "result",
            item,
        )

        bbox = (
            item.get("detection_bbox")
            or item.get("bbox")
            or crop.get("detection_bbox")
            or crop.get("bbox")
        )

        if bbox is None:
            bbox_xyxy = (
                item.get("bbox_xyxy")
                or crop.get("bbox_xyxy")
            )

            if isinstance(
                bbox_xyxy,
                dict,
            ):
                bbox = [
                    bbox_xyxy.get("x1"),
                    bbox_xyxy.get("y1"),
                    bbox_xyxy.get("x2"),
                    bbox_xyxy.get("y2"),
                ]

        if (
            bbox is None
            or len(bbox) != 4
            or any(
                value is None
                for value in bbox
            )
        ):
            return None

        predictions = (
            item.get("predictions")
            or result.get("predictions")
            or []
        )

        primary = (
            self._select_primary_label(
                predictions
            )
        )

        detector_score = (
            item.get(
                "detector_confidence"
            )
        )

        if detector_score is None:
            detector_score = item.get(
                "detector_score"
            )

        if detector_score is None:
            detector_score = crop.get(
                "detector_confidence"
            )

        if detector_score is None:
            detector_score = crop.get(
                "detector_score"
            )

        if primary is None:
            mapped_labels = (
                item.get(
                    "mapped_labels"
                )
                or crop.get(
                    "mapped_labels"
                )
                or []
            )

            if mapped_labels:
                primary = {
                    "label": str(
                        mapped_labels[0]
                    ),

                    "score": float(
                        detector_score
                        or 0.0
                    ),
                }

            else:
                detector_label = (
                    item.get(
                        "detector_label"
                    )
                    or crop.get(
                        "detector_label"
                    )
                )

                if detector_label:
                    primary = {
                        "label": str(
                            detector_label
                        ),

                        "score": float(
                            detector_score
                            or 0.0
                        ),
                    }

        if primary is None:
            return None

        crop_path = (
            item.get("crop_path")
            or crop.get("crop_path")
            or result.get("frame_path")
        )

        return {
            "detection_id": (
                f"frame_{frame_id:06d}_"
                f"object_{index:02d}"
            ),

            "frame_id": frame_id,

            "timestamp": timestamp,

            "scene_id": scene_id,

            "label": primary[
                "label"
            ],

            "label_score": float(
                primary["score"]
            ),

            "bbox": [
                float(value)
                for value in bbox
            ],

            "crop_path": crop_path,

            "detector_confidence": (
                float(detector_score)
                if detector_score
                is not None
                else None
            ),
        }

    def _select_primary_label(
        self,
        predictions,
    ):
        valid = []

        for prediction in predictions:
            label = prediction.get(
                "label"
            )

            score = prediction.get(
                "score"
            )

            if (
                label is None
                or score is None
            ):
                continue

            valid.append(
                {
                    "label": str(
                        label
                    ),

                    "score": float(
                        score
                    ),
                }
            )

        if not valid:
            return None

        specific = [
            prediction
            for prediction in valid
            if prediction["label"]
            not in self.GENERIC_LABELS
        ]

        candidates = (
            specific
            if specific
            else valid
        )

        return max(
            candidates,
            key=lambda item: item[
                "score"
            ],
        )