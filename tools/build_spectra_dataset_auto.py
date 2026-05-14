import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ai.spectra.label_sets import (
    OBJECT_LABELS,
    ACTION_LABELS,
    SCENARIO_LABELS,
    COMPOSITION_LABELS,
    SPECTRA_LABELS,
)


PIXABAY_VIDEO_API_URL = "https://pixabay.com/api/videos/"


LABEL_PROMPTS = {
    "person": "a person in the image",
    "book": "a book in the image",
    "table": "a table in the image",
    "chair": "a chair in the image",
    "sofa": "a sofa in the image",
    "door": "a door in the image",
    "window": "a window in the image",
    "phone": "a phone in the image",
    "computer": "a computer in the image",
    "screen": "a screen in the image",
    "car": "a car in the image",
    "animal": "an animal in the image",
    "food": "food in the image",
    "cup": "a cup in the image",
    "bag": "a bag in the image",
    "weapon": "a weapon in the image",
    "knife": "a knife in the image",
    "dice": "dice in the image",
    "miniature": "a miniature figure in the image",
    "board_game": "a board game on a table",
    "musical_instrument": "a musical instrument in the image",
    "cap": "a cap or hat in the image",
    "subtitles": "subtitles at the bottom of the video frame",
    "on_screen_text": "text displayed on the screen",

    "sitting": "a person sitting",
    "standing": "a person standing",
    "walking": "a person walking",
    "running": "a person running",
    "talking": "a person talking",
    "looking": "a person looking at something",
    "holding": "a person holding an object",
    "reading": "a person reading",
    "writing": "a person writing",
    "playing": "a person playing",
    "fighting": "people fighting",
    "falling": "a person falling",
    "pointing": "a person pointing",
    "smiling": "a person smiling",
    "opening": "a person opening something",
    "showing": "a person showing something",

    "indoor": "an indoor scene",
    "outdoor": "an outdoor scene",
    "room": "a room",
    "street": "a street",
    "school": "a school",
    "kitchen": "a kitchen",
    "bedroom": "a bedroom",
    "living_room": "a living room",
    "forest": "a forest",
    "city": "a city",
    "sports_field": "a sports field",
    "dark_place": "a dark place",
    "bright_place": "a bright place",
    "day": "daytime",
    "night": "nighttime",

    "close_up": "a close up shot",
    "medium_shot": "a medium shot",
    "wide_shot": "a wide shot",
    "one_person": "one person in the image",
    "two_people": "two people in the image",
    "group_of_people": "a group of people",
    "crowded_scene": "a crowded scene",
    "empty_scene": "an empty scene with no people",
}


LABEL_GROUPS = {
    "objects": OBJECT_LABELS,
    "actions": ACTION_LABELS,
    "scenarios": SCENARIO_LABELS,
    "composition": COMPOSITION_LABELS,
}


GROUP_TOP_K = {
    "objects": 6,
    "actions": 4,
    "scenarios": 4,
    "composition": 3,
}


GROUP_MIN_SCORE = {
    "objects": 0.08,
    "actions": 0.10,
    "scenarios": 0.10,
    "composition": 0.12,
}


def sanitize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def search_pixabay_videos(
    api_key: str,
    query: str,
    per_page: int,
    safesearch: bool = True,
) -> List[dict]:
    per_page = max(3, min(per_page, 200))

    params = {
        "key": api_key,
        "q": query,
        "per_page": per_page,
        "safesearch": "true" if safesearch else "false",
        "video_type": "film",
    }

    response = requests.get(
        PIXABAY_VIDEO_API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data.get("hits", [])


def choose_video_url(video_hit: dict, quality: str = "small") -> str:
    videos = video_hit.get("videos", {})

    preferred_order = [quality, "small", "tiny", "medium", "large"]

    for key in preferred_order:
        video_data = videos.get(key)

        if video_data and video_data.get("url"):
            return video_data["url"]

    raise ValueError("Nenhuma URL de vídeo encontrada no resultado da Pixabay.")


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def extract_spaced_frames(
    video_path: Path,
    frames_dir: Path,
    interval_seconds: float,
    max_frames: int,
) -> List[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError(f"FPS inválido no vídeo: {video_path}")

    frame_step = max(1, int(fps * interval_seconds))

    saved_frames = []
    current_frame = 0
    saved_count = 0

    while current_frame < total_frames and saved_count < max_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        success, frame = capture.read()

        if not success:
            break

        timestamp = current_frame / fps
        frame_path = frames_dir / f"frame_{saved_count:06d}_t{timestamp:.2f}.jpg"

        cv2.imwrite(str(frame_path), frame)

        saved_frames.append(frame_path)

        saved_count += 1
        current_frame += frame_step

    capture.release()

    return saved_frames


class ClipAutoLabeler:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.model.eval()

        self.text_features_by_group = self._build_text_features()

    def _build_text_features(self) -> Dict[str, Tuple[List[str], torch.Tensor]]:
        text_features_by_group = {}

        with torch.no_grad():
            for group_name, labels in LABEL_GROUPS.items():
                prompts = [
                    LABEL_PROMPTS.get(label, label.replace("_", " "))
                    for label in labels
                ]

                inputs = self.processor(
                    text=prompts,
                    return_tensors="pt",
                    padding=True
                )

                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }

                text_features = self.model.get_text_features(**inputs)

                text_features = text_features / text_features.norm(
                    dim=-1,
                    keepdim=True
                )

                text_features_by_group[group_name] = (labels, text_features)

        return text_features_by_group

    @torch.no_grad()
    def label_image(self, image_path: Path) -> Tuple[Dict[str, int], Dict[str, float]]:
        image = Image.open(image_path).convert("RGB")

        image_inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        image_inputs = {
            key: value.to(self.device)
            for key, value in image_inputs.items()
        }

        image_features = self.model.get_image_features(**image_inputs)

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        binary_labels = {
            label: 0
            for label in SPECTRA_LABELS
        }

        score_labels = {
            label: 0.0
            for label in SPECTRA_LABELS
        }

        for group_name, (labels, text_features) in self.text_features_by_group.items():
            similarities = image_features @ text_features.T

            probabilities = similarities.softmax(dim=-1).squeeze(0).cpu()

            scored_labels = []

            for label, probability in zip(labels, probabilities):
                score = float(probability)
                score_labels[label] = round(score, 4)

                scored_labels.append((label, score))

            scored_labels.sort(
                key=lambda item: item[1],
                reverse=True
            )

            top_k = GROUP_TOP_K[group_name]
            min_score = GROUP_MIN_SCORE[group_name]

            for label, score in scored_labels[:top_k]:
                if score >= min_score:
                    binary_labels[label] = 1

        return binary_labels, score_labels


def build_auto_dataset(
    queries: List[str],
    videos_per_query: int,
    frames_per_video: int,
    interval_seconds: float,
    output_dir: Path,
    output_csv: Path,
    output_scores_csv: Path,
    video_quality: str,
) -> None:
    api_key = os.getenv("PIXABAY_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "A variável de ambiente PIXABAY_API_KEY não foi definida."
        )

    labeler = ClipAutoLabeler()

    rows = []
    score_rows = []

    for query in queries:
        print(f"\nBuscando vídeos para: {query}")

        hits = search_pixabay_videos(
            api_key=api_key,
            query=query,
            per_page=videos_per_query,
        )

        for index, hit in enumerate(hits):
            video_id = hit["id"]
            safe_query = sanitize_name(query)

            video_dir = output_dir / safe_query / f"video_{video_id}"
            video_path = video_dir / f"video_{video_id}.mp4"
            frames_dir = video_dir / "frames"

            video_url = choose_video_url(
                video_hit=hit,
                quality=video_quality
            )

            print(f"Baixando vídeo {video_id}...")
            download_file(video_url, video_path)

            print("Extraindo frames...")
            frame_paths = extract_spaced_frames(
                video_path=video_path,
                frames_dir=frames_dir,
                interval_seconds=interval_seconds,
                max_frames=frames_per_video,
            )

            for frame_path in frame_paths:
                print(f"Rotulando: {frame_path}")

                labels, scores = labeler.label_image(frame_path)

                row = {
                    "frame_path": str(frame_path),
                    "source": "pixabay",
                    "source_query": query,
                    "source_video_id": video_id,
                    "source_page_url": hit.get("pageURL", ""),
                }

                score_row = dict(row)

                for label in SPECTRA_LABELS:
                    row[label] = labels[label]
                    score_row[label] = scores[label]

                rows.append(row)
                score_rows.append(score_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    pd.DataFrame(score_rows).to_csv(output_scores_csv, index=False)

    print(f"\nCSV de labels criado em: {output_csv}")
    print(f"CSV de scores criado em: {output_scores_csv}")
    print(f"Total de frames: {len(rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Baixa vídeos da Pixabay, extrai frames e cria labels automáticas com CLIP."
    )

    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "person walking",
            "person talking",
            "running",
            "kitchen",
            "city street",
            "forest",
            "sports",
            "computer",
        ],
        help="Termos de busca para vídeos."
    )

    parser.add_argument(
        "--videos-per-query",
        type=int,
        default=2,
        help="Quantidade de vídeos por termo de busca."
    )

    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=6,
        help="Quantidade máxima de frames por vídeo."
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Intervalo em segundos entre frames."
    )

    parser.add_argument(
        "--output-dir",
        default="data/dataset_sources/auto_pixabay",
        help="Pasta onde vídeos e frames serão salvos."
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/spectra_auto_labels.csv",
        help="CSV de saída com labels 0/1."
    )

    parser.add_argument(
        "--output-scores-csv",
        default="data/datasets/spectra_auto_scores.csv",
        help="CSV de saída com scores do CLIP."
    )

    parser.add_argument(
        "--video-quality",
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Qualidade do vídeo baixado."
    )

    args = parser.parse_args()

    build_auto_dataset(
        queries=args.queries,
        videos_per_query=args.videos_per_query,
        frames_per_video=args.frames_per_video,
        interval_seconds=args.interval,
        output_dir=Path(args.output_dir),
        output_csv=Path(args.output_csv),
        output_scores_csv=Path(args.output_scores_csv),
        video_quality=args.video_quality,
    )


if __name__ == "__main__":
    main()