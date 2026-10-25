
import os
import time
import threading
from functools import lru_cache

import cv2
import numpy as np
import torch

from transformers import (
    XCLIPModel,
    XCLIPProcessor,
)

from ultralytics import YOLO


MODEL_ID = "microsoft/xclip-base-patch32-16-frames"

ACTION_PROMPTS = {
    "motion": {
        "walking": "a video of a person walking on foot",
        "running": "a video of a person running",
        "standing": "a video of a person standing still",
        "sitting": "a video of a person sitting down",
        "lying_down": "a video of a person lying down",
        "crouching": "a video of a person crouching",
        "jumping": "a video of a person jumping",
        "dancing": "a video of a person dancing",
        "cycling": "a video of a person riding a bicycle",
        "climbing": "a video of a person climbing",
        "swimming": "a video of a person swimming",
        "falling": "a video of a person falling down",
    },
    "activity": {
        "phone_use": "a video of a person using a mobile phone",
        "talking": "a video of a person talking to someone",
        "reading": "a video of a person reading a book",
        "writing": "a video of a person writing",
        "eating": "a video of a person eating food",
        "drinking": "a video of a person drinking",
        "computer_use": "a video of a person using a computer",
        "instrument_playing": "a video of a person playing a musical instrument",
        "cooking": "a video of a person cooking food",
        "cleaning": "a video of a person cleaning",
        "carrying": "a video of a person carrying an object",
        "reaching": "a video of a person reaching for something",
        "throwing": "a video of a person throwing an object",
        "driving": "a video of a person driving a car",
        "exercising": "a video of a person exercising",
        "working": "a video of a person doing manual work",
        "playing": "a video of a person playing a game",
    },
}


class XCLIPVerifier:
    NUM_FRAMES = 16

    def __init__(
        self,
        model_id=MODEL_ID,
        device=None,
        window_seconds=4.0,
        max_people=2,
        person_confidence=0.35,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.window_seconds = float(window_seconds)
        self.max_people = int(max_people)
        self.person_confidence = float(person_confidence)
        self.lock = threading.RLock()

        self.entries = []

        for group, prompts in ACTION_PROMPTS.items():
            for label, prompt in prompts.items():
                self.entries.append({
                    "group": group,
                    "label": label,
                    "prompt": prompt,
                })

        self.prompts = [
            entry["prompt"]
            for entry in self.entries
        ]

        print("[XCLIP] Carregando:", model_id)
        start = time.perf_counter()

        self.processor = XCLIPProcessor.from_pretrained(
            model_id
        )

        self.model = XCLIPModel.from_pretrained(
            model_id,
            use_safetensors=True,
        )

        self.model = self.model.to(
            self.device
        ).eval()

        self.detector = None

        if self.max_people > 0:
            self.detector = YOLO(
                "yolov8n.pt"
            )

        self.load_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        print(
            "[XCLIP] Pronto.",
            "Device:", self.device,
            "Carregamento:", self.load_ms, "ms",
        )

    def _sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def _decode_clip(
        self,
        capture,
        timestamp,
        scene_start,
        scene_end,
    ):
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0 or total <= 0:
            raise RuntimeError(
                "Vídeo com FPS ou quantidade "
                "de frames inválidos."
            )

        video_end = (total - 1) / fps

        start_limit = max(
            0.0,
            float(scene_start),
        )

        end_limit = min(
            video_end,
            float(scene_end),
        )

        if end_limit < start_limit:
            end_limit = start_limit

        length = min(
            self.window_seconds,
            end_limit - start_limit,
        )

        start = max(
            start_limit,
            min(
                float(timestamp) - length / 2,
                end_limit - length,
            ),
        )

        end = start + length

        frame_indices = np.rint(
            np.linspace(
                start * fps,
                end * fps,
                self.NUM_FRAMES,
            )
        ).astype(int)

        frame_indices = np.clip(
            frame_indices,
            0,
            total - 1,
        )

        first_index = int(frame_indices[0])

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            first_index,
        )

        frames = []
        current = first_index
        previous_index = None
        previous_frame = None

        for raw_index in frame_indices:
            index = int(raw_index)

            if index == previous_index:
                frames.append(
                    previous_frame.copy()
                )
                continue

            while current < index:
                if not capture.grab():
                    raise RuntimeError(
                        "Falha ao avançar no vídeo."
                    )

                current += 1

            success, frame = capture.read()

            if not success:
                raise RuntimeError(
                    f"Falha ao ler frame {index}."
                )

            frames.append(frame)

            current += 1
            previous_index = index
            previous_frame = frame

        return frames, start, end

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        intersection = max(
            0.0, x2 - x1
        ) * max(
            0.0, y2 - y1
        )

        area_a = max(
            0.0, a[2] - a[0]
        ) * max(
            0.0, a[3] - a[1]
        )

        area_b = max(
            0.0, b[2] - b[0]
        ) * max(
            0.0, b[3] - b[1]
        )

        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def _detect_people(self, frame):
        if self.detector is None:
            return []

        results = self.detector.predict(
            source=frame,
            classes=[0],
            conf=self.person_confidence,
            verbose=False,
            device=(
                0 if self.device.startswith("cuda")
                else "cpu"
            ),
        )

        people = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                coordinates = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                confidence = float(
                    box.conf[0]
                    .detach()
                    .cpu()
                )

                people.append({
                    "bbox": coordinates,
                    "confidence": confidence,
                })

        return sorted(
            people,
            key=lambda p: p["confidence"],
            reverse=True,
        )

    def _select_nearest(self, previous, detections, shape):
        if not detections:
            return previous

        height, width = shape[:2]
        diagonal = max(
            (width ** 2 + height ** 2) ** 0.5,
            1.0,
        )

        def rank(item):
            box = item["bbox"]

            previous_center = (
                (previous[0] + previous[2]) / 2,
                (previous[1] + previous[3]) / 2,
            )

            current_center = (
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2,
            )

            distance = (
                (
                    current_center[0] - previous_center[0]
                ) ** 2
                + (
                    current_center[1] - previous_center[1]
                ) ** 2
            ) ** 0.5 / diagonal

            overlap = self._iou(
                previous,
                box,
            )

            return overlap - 0.4 * distance

        selected = max(
            detections,
            key=rank,
        )["bbox"]

        if rank({"bbox": selected}) < -0.10:
            return previous

        return selected

    def _person_views(self, frames):
        if self.detector is None:
            return []

        count = len(frames)
        anchor_index = count // 2

        keyframes = sorted(
            set(
                [0, count - 1, anchor_index]
                + list(range(0, count, 4))
            )
        )

        detections = {
            index: self._detect_people(
                frames[index]
            )
            for index in keyframes
        }

        anchors = detections.get(
            anchor_index,
            [],
        )[:self.max_people]

        tracks = []

        for person_index, anchor in enumerate(anchors):
            anchor_box = anchor["bbox"]

            known = {
                anchor_index: anchor_box
            }

            for direction in (-1, 1):
                previous = anchor_box

                indices = [
                    index
                    for index in keyframes
                    if (
                        index < anchor_index
                        if direction == -1
                        else index > anchor_index
                    )
                ]

                if direction == -1:
                    indices.reverse()

                for index in indices:
                    previous = self._select_nearest(
                        previous,
                        detections[index],
                        frames[index].shape,
                    )

                    known[index] = previous

            known_indices = sorted(
                known.keys()
            )

            interpolated = []

            for frame_index in range(count):
                interpolated.append([
                    float(
                        np.interp(
                            frame_index,
                            known_indices,
                            [
                                known[index][coordinate]
                                for index in known_indices
                            ],
                        )
                    )
                    for coordinate in range(4)
                ])

            person_frames = []

            for frame, bbox in zip(
                frames,
                interpolated,
            ):
                height, width = frame.shape[:2]

                x1, y1, x2, y2 = bbox

                padding_x = (
                    x2 - x1
                ) * 0.20

                padding_y = (
                    y2 - y1
                ) * 0.20

                left = max(
                    0,
                    int(x1 - padding_x),
                )

                top = max(
                    0,
                    int(y1 - padding_y),
                )

                right = min(
                    width,
                    int(x2 + padding_x),
                )

                bottom = min(
                    height,
                    int(y2 + padding_y),
                )

                if right <= left or bottom <= top:
                    crop = frame
                else:
                    crop = frame[
                        top:bottom,
                        left:right,
                    ]

                person_frames.append(
                    crop
                )

            tracks.append({
                "person_index": person_index,
                "anchor_bbox": anchor_box,
                "frames": person_frames,
                "tracking_method": (
                    "keyframe_yolo_iou_interpolation"
                ),
            })

        return tracks

    def _predict_view(
        self,
        frames,
        view_name,
        person_index=None,
    ):
        rgb_frames = [
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
            for frame in frames
        ]

        inputs = self.processor(
            text=self.prompts,
            videos=rgb_frames,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        self._sync()
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model(
                **inputs
            )

        self._sync()

        inference_ms = (
            time.perf_counter() - start
        ) * 1000

        logits = (
            outputs.logits_per_video[0]
            .detach()
            .float()
            .cpu()
        )

        groups = {}

        for group in ACTION_PROMPTS:
            indices = [
                index
                for index, entry in enumerate(
                    self.entries
                )
                if entry["group"] == group
            ]

            group_logits = logits[indices]

            relative_scores = torch.softmax(
                group_logits,
                dim=0,
            )

            predictions = []

            for local_index, global_index in enumerate(
                indices
            ):
                predictions.append({
                    "label": self.entries[
                        global_index
                    ]["label"],
                    "relative_score": round(
                        float(
                            relative_scores[local_index]
                        ),
                        6,
                    ),
                    "logit": round(
                        float(
                            group_logits[local_index]
                        ),
                        5,
                    ),
                })

            predictions.sort(
                key=lambda item: item[
                    "relative_score"
                ],
                reverse=True,
            )

            groups[group] = predictions

        return {
            "view": view_name,
            "person_index": person_index,
            "groups": groups,
            "inference_ms": round(
                inference_ms,
                2,
            ),
        }

    def predict_clip(
        self,
        capture,
        timestamp,
        scene_start,
        scene_end,
    ):
        with self.lock:
            started = time.perf_counter()

            frames, start, end = self._decode_clip(
                capture,
                timestamp,
                scene_start,
                scene_end,
            )

            views = []

            views.append(
                self._predict_view(
                    frames,
                    "full_frame",
                )
            )

            people = self._person_views(
                frames
            )

            for person in people:
                view = self._predict_view(
                    person["frames"],
                    "person",
                    person["person_index"],
                )

                view["anchor_bbox"] = (
                    person["anchor_bbox"]
                )

                view["tracking_method"] = (
                    person["tracking_method"]
                )

                views.append(view)

            total_ms = (
                time.perf_counter()
                - started
            ) * 1000

            return {
                "model": MODEL_ID,
                "timestamp": float(timestamp),
                "window_start": round(start, 3),
                "window_end": round(end, 3),
                "frames_analyzed": len(frames),
                "people_detected": len(people),
                "views": views,
                "total_ms": round(
                    total_ms,
                    2,
                ),
            }


@lru_cache(maxsize=1)
def get_xclip_verifier():
    return XCLIPVerifier(
        window_seconds=float(
            os.getenv(
                "SEE2SOUND_XCLIP_WINDOW_SECONDS",
                "4.0",
            )
        ),
        max_people=int(
            os.getenv(
                "SEE2SOUND_XCLIP_MAX_PEOPLE",
                "2",
            )
        ),
    )
