import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ai.spectra.labels.label_sets import (
    SPECTRA_LABELS,
    SPECTRA_LABEL_GROUPS,
)


PIXABAY_VIDEO_API_URL = "https://pixabay.com/api/videos/"


LABEL_PROMPTS = {
    # Person labels
    "person": "a person in the image",
    "face": "a human face visible in the image",
    "hand": "a human hand visible in the image",
    "man": "a man in the image",
    "woman": "a woman in the image",
    "child": "a child in the image",
    "group_of_people": "a group of people in the image",

    # Common objects
    "book": "a book in the image",
    "table": "a table in the image",
    "chair": "a chair in the image",
    "sofa": "a sofa in the image",
    "bed": "a bed in the image",
    "door": "a door in the image",
    "window": "a window in the image",
    "phone": "a phone in the image",
    "computer": "a computer in the image",
    "screen": "a screen in the image",
    "television": "a television in the image",
    "car": "a car in the image",
    "bicycle": "a bicycle in the image",
    "motorcycle": "a motorcycle in the image",
    "bus": "a bus in the image",
    "animal": "an animal in the image",
    "dog": "a dog in the image",
    "cat": "a cat in the image",
    "food": "food in the image",
    "cup": "a cup in the image",
    "bottle": "a bottle in the image",
    "bag": "a bag in the image",
    "backpack": "a backpack in the image",
    "weapon": "a weapon in the image",
    "knife": "a knife in the image",
    "ball": "a ball in the image",
    "toy": "a toy in the image",
    "paper": "a sheet of paper in the image",
    "box": "a box in the image",

    # Specific objects
    "dice": "dice in the image",
    "miniature": "a small miniature figure in the image",
    "board_game": "a board game on a table",
    "musical_instrument": "a musical instrument in the image",
    "cap": "a cap or hat in the image",
    "subtitles": "subtitles at the bottom of the video frame",
    "on_screen_text": "text displayed on the screen",

    # Actions
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
    "jumping": "a person jumping",
    "pointing": "a person pointing",
    "smiling": "a person smiling",
    "crying": "a person crying",
    "opening": "a person opening something",
    "closing": "a person closing something",
    "showing": "a person showing something",
    "eating": "a person eating",
    "drinking": "a person drinking",
    "driving": "a person driving",
    "dancing": "a person dancing",
    "working": "a person working",

    # Scenarios
    "indoor": "an indoor scene",
    "outdoor": "an outdoor scene",
    "room": "a room",
    "street": "a street",
    "school": "a school",
    "classroom": "a classroom",
    "kitchen": "a kitchen",
    "bedroom": "a bedroom",
    "living_room": "a living room",
    "office": "an office",
    "forest": "a forest",
    "city": "a city",
    "sports_field": "a sports field",
    "beach": "a beach",
    "park": "a park",
    "store": "a store",
    "restaurant": "a restaurant",
    "hospital": "a hospital",
    "dark_place": "a dark place",
    "bright_place": "a bright place",
    "day": "daytime scene",
    "night": "nighttime scene",

    # Composition
    "close_up": "a close up shot",
    "medium_shot": "a medium shot",
    "wide_shot": "a wide shot",
    "one_person": "one person in the image",
    "two_people": "two people in the image",
    "crowded_scene": "a crowded scene",
    "empty_scene": "an empty scene with no people",

    # Visual state
    "calm_scene": "a calm scene",
    "action_scene": "an action scene",
    "conversation_scene": "a conversation scene",
    "movement_scene": "a scene with movement",

    # Hair labels, caso você tenha adicionado no label_sets.py
    "blonde_hair": "a person with blonde hair",
    "brown_hair": "a person with brown hair",
    "black_hair": "a person with black hair",
    "red_hair": "a person with red hair",
    "gray_hair": "a person with gray hair",
    "short_hair": "a person with short hair",
    "long_hair": "a person with long hair",
    "curly_hair": "a person with curly hair",
    "straight_hair": "a person with straight hair",

    # Clothing labels, caso você tenha adicionado no label_sets.py
    "red_clothes": "a person wearing red clothes",
    "blue_clothes": "a person wearing blue clothes",
    "black_clothes": "a person wearing black clothes",
    "white_clothes": "a person wearing white clothes",
    "green_clothes": "a person wearing green clothes",
    "yellow_clothes": "a person wearing yellow clothes",
    "dress": "a person wearing a dress",
    "shirt": "a person wearing a shirt",
    "jacket": "a person wearing a jacket",
    "hat": "a person wearing a hat",
    "glasses": "a person wearing glasses",

    # Skin tone labels, caso você tenha adicionado no label_sets.py
    "light_skin": "a person with light skin",
    "medium_skin": "a person with medium skin tone",
    "dark_skin": "a person with dark skin",
}


DEFAULT_GROUP_THRESHOLDS = {
    "person": 0.52,
    "common_objects": 0.54,
    "specific_objects": 0.54,
    "actions": 0.53,
    "scenarios": 0.53,
    "composition": 0.53,
    "visual_state": 0.54,

    "hair": 0.56,
    "clothing": 0.56,
    "skin_tone": 0.56,
}

GROUP_NONE_PROMPTS = {
    "person": "a photo with no people",
    "common_objects": "a photo with no relevant objects",
    "specific_objects": "a photo without special objects",
    "actions": "a photo with no human action",
    "scenarios": "a photo with no specific place",
    "composition": "a generic photo",
    "visual_state": "a neutral scene",

    "hair": "a photo with no visible hair details",
    "clothing": "a photo with no visible clothing details",
    "skin_tone": "a photo with no visible skin tone details",
}


DEFAULT_GROUP_TOP_K = {
    "person": 4,
    "common_objects": 6,
    "specific_objects": 4,
    "actions": 4,
    "scenarios": 4,
    "composition": 3,
    "visual_state": 2,

    "hair": 2,
    "clothing": 3,
    "skin_tone": 1,
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
    requested_per_page = per_page
    api_per_page = max(3, min(per_page, 200))

    params = {
        "key": api_key,
        "q": query,
        "per_page": api_per_page,
        "safesearch": "true" if safesearch else "false",
        "video_type": "film",
    }

    response = requests.get(
        PIXABAY_VIDEO_API_URL,
        params=params,
        timeout=30
    )

    if response.status_code >= 400:
        print("Erro ao consultar Pixabay:")
        print(response.status_code)
        print(response.text)

    response.raise_for_status()

    data = response.json()

    return data.get("hits", [])[:requested_per_page]


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


def get_prompt_for_label(label: str) -> str:
    return LABEL_PROMPTS.get(label, label.replace("_", " "))


class ClipAutoLabeler:
    def __init__(
        self,
        device: Optional[str] = None,
        global_threshold: Optional[float] = None,
        use_top_k: bool = True,
        margin_scale: float = 50.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.global_threshold = global_threshold
        self.use_top_k = use_top_k
        self.margin_scale = margin_scale

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

        self.model.eval()

        self.text_features_by_group = self._build_group_text_features()

    def _build_group_text_features(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Cria embeddings de texto por grupo.

        Para cada grupo, cria:
        - embeddings das labels
        - embedding de uma opção "none" / ausência
        """
        text_features_by_group = {}

        with torch.no_grad():
            for group_name, labels in SPECTRA_LABEL_GROUPS.items():
                label_prompts = [
                    get_prompt_for_label(label)
                    for label in labels
                ]

                none_prompt = GROUP_NONE_PROMPTS.get(
                    group_name,
                    "a generic photo"
                )

                label_inputs = self.processor(
                    text=label_prompts,
                    return_tensors="pt",
                    padding=True
                )

                label_inputs = {
                    key: value.to(self.device)
                    for key, value in label_inputs.items()
                }

                label_features = self.model.get_text_features(**label_inputs)

                label_features = label_features / label_features.norm(
                    dim=-1,
                    keepdim=True
                )

                none_inputs = self.processor(
                    text=[none_prompt],
                    return_tensors="pt",
                    padding=True
                )

                none_inputs = {
                    key: value.to(self.device)
                    for key, value in none_inputs.items()
                }

                none_features = self.model.get_text_features(**none_inputs)

                none_features = none_features / none_features.norm(
                    dim=-1,
                    keepdim=True
                )

                text_features_by_group[group_name] = {
                    "labels": labels,
                    "label_features": label_features,
                    "none_features": none_features,
                }

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

        for group_name, group_data in self.text_features_by_group.items():
            labels = group_data["labels"]
            label_features = group_data["label_features"]
            none_features = group_data["none_features"]

            label_similarities = (image_features @ label_features.T).squeeze(0).cpu()
            none_similarity = float((image_features @ none_features.T).squeeze().cpu())

            scored_labels = []

            for label, label_similarity in zip(labels, label_similarities):
                label_similarity = float(label_similarity)

                margin = label_similarity - none_similarity

                confidence = torch.sigmoid(
                    torch.tensor(margin * self.margin_scale)
                ).item()

                score_labels[label] = round(confidence, 4)

                scored_labels.append((label, confidence))

            scored_labels.sort(
                key=lambda item: item[1],
                reverse=True
            )

            threshold = self._get_threshold_for_group(group_name)
            top_k = self._get_top_k_for_group(group_name)

            if self.use_top_k:
                candidates = scored_labels[:top_k]
            else:
                candidates = scored_labels

            for label, confidence in candidates:
                if confidence > threshold:
                    binary_labels[label] = 1
                else:
                    binary_labels[label] = 0

        self._apply_consistency_rules(binary_labels, score_labels)

        return binary_labels, score_labels

    def _get_threshold_for_group(self, group_name: str) -> float:
        if self.global_threshold is not None:
            return self.global_threshold

        return DEFAULT_GROUP_THRESHOLDS.get(group_name, 0.53)

    def _get_top_k_for_group(self, group_name: str) -> int:
        return DEFAULT_GROUP_TOP_K.get(group_name, 4)

    def _apply_consistency_rules(self, binary_labels: Dict[str, int], score_labels: Dict[str, float]) -> None:
        """
        Aplica regras simples para reduzir labels contraditórias.
        """

        def keep_only_best(labels):
            existing_labels = [
                label for label in labels
                if label in binary_labels and binary_labels[label] == 1
            ]

            if len(existing_labels) <= 1:
                return

            best_label = max(
                existing_labels,
                key=lambda label: score_labels.get(label, 0.0)
            )

            for label in existing_labels:
                binary_labels[label] = 1 if label == best_label else 0

        # Grupos mutuamente exclusivos.
        keep_only_best([
            "one_person",
            "two_people",
            "group_of_people",
            "crowded_scene",
            "empty_scene",
        ])

        keep_only_best([
            "close_up",
            "medium_shot",
            "wide_shot",
        ])

        keep_only_best([
            "calm_scene",
            "action_scene",
            "conversation_scene",
            "movement_scene",
        ])

        keep_only_best([
            "indoor",
            "outdoor",
        ])

        keep_only_best([
            "day",
            "night",
        ])

        keep_only_best([
            "light_skin",
            "medium_skin",
            "dark_skin",
        ])

        # Se empty_scene venceu, remove pessoas e atributos humanos.
        if binary_labels.get("empty_scene", 0) == 1:
            person_related_labels = [
                "person",
                "face",
                "hand",
                "man",
                "woman",
                "child",
                "group_of_people",
                "one_person",
                "two_people",
                "crowded_scene",

                "sitting",
                "standing",
                "walking",
                "running",
                "talking",
                "looking",
                "holding",
                "reading",
                "writing",
                "playing",
                "fighting",
                "falling",
                "jumping",
                "pointing",
                "smiling",
                "crying",
                "opening",
                "closing",
                "showing",
                "eating",
                "drinking",
                "driving",
                "dancing",
                "working",

                "blonde_hair",
                "brown_hair",
                "black_hair",
                "red_hair",
                "gray_hair",
                "short_hair",
                "long_hair",
                "curly_hair",
                "straight_hair",

                "red_clothes",
                "blue_clothes",
                "black_clothes",
                "white_clothes",
                "green_clothes",
                "yellow_clothes",
                "dress",
                "shirt",
                "jacket",
                "hat",
                "glasses",

                "light_skin",
                "medium_skin",
                "dark_skin",
            ]

            for label in person_related_labels:
                if label in binary_labels:
                    binary_labels[label] = 0

        # Se não há pessoa, remove ações e aparência.
        if binary_labels.get("person", 0) == 0:
            person_dependent_labels = [
                "face",
                "hand",
                "man",
                "woman",
                "child",
                "group_of_people",
                "one_person",
                "two_people",

                "sitting",
                "standing",
                "walking",
                "running",
                "talking",
                "looking",
                "holding",
                "reading",
                "writing",
                "playing",
                "fighting",
                "falling",
                "jumping",
                "pointing",
                "smiling",
                "crying",
                "opening",
                "closing",
                "showing",
                "eating",
                "drinking",
                "driving",
                "dancing",
                "working",

                "blonde_hair",
                "brown_hair",
                "black_hair",
                "red_hair",
                "gray_hair",
                "short_hair",
                "long_hair",
                "curly_hair",
                "straight_hair",

                "red_clothes",
                "blue_clothes",
                "black_clothes",
                "white_clothes",
                "green_clothes",
                "yellow_clothes",
                "dress",
                "shirt",
                "jacket",
                "hat",
                "glasses",

                "light_skin",
                "medium_skin",
                "dark_skin",
            ]

            for label in person_dependent_labels:
                if label in binary_labels:
                    binary_labels[label] = 0

        # Se tem man/woman/child, garante person.
        if (
            binary_labels.get("man", 0) == 1
            or binary_labels.get("woman", 0) == 1
            or binary_labels.get("child", 0) == 1
            or binary_labels.get("face", 0) == 1
        ):
            if "person" in binary_labels:
                binary_labels["person"] = 1

    def _get_threshold_for_group(self, group_name: str) -> float:
        if self.global_threshold is not None:
            return self.global_threshold

        return DEFAULT_GROUP_THRESHOLDS.get(group_name, 0.12)

    def _get_top_k_for_group(self, group_name: str) -> int:
        return DEFAULT_GROUP_TOP_K.get(group_name, 5)


def build_auto_dataset(
    queries: List[str],
    videos_per_query: int,
    frames_per_video: int,
    interval_seconds: float,
    output_dir: Path,
    output_csv: Path,
    output_scores_csv: Path,
    video_quality: str,
    global_threshold: Optional[float],
    use_top_k: bool,
    margin_scale: float,
) -> None:
    api_key = os.getenv("PIXABAY_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "A variável de ambiente PIXABAY_API_KEY não foi definida."
        )

    labeler = ClipAutoLabeler(
        global_threshold=global_threshold,
        use_top_k=use_top_k,
        margin_scale=margin_scale,
    )

    rows = []
    score_rows = []

    for query in queries:
        print(f"\nBuscando vídeos para: {query}")

        hits = search_pixabay_videos(
            api_key=api_key,
            query=query,
            per_page=videos_per_query,
        )

        for hit in hits:
            video_id = hit["id"]
            safe_query = sanitize_name(query)

            video_dir = output_dir / safe_query / f"video_{video_id}"
            video_path = video_dir / f"video_{video_id}.mp4"
            frames_dir = video_dir / "frames"

            video_url = choose_video_url(
                video_hit=hit,
                quality=video_quality
            )

            if not video_path.exists():
                print(f"Baixando vídeo {video_id}...")
                download_file(video_url, video_path)
            else:
                print(f"Vídeo {video_id} já existe. Pulando download.")

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
                    row[label] = int(labels[label])
                    score_row[label] = scores[label]

                rows.append(row)
                score_rows.append(score_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_scores_csv.parent.mkdir(parents=True, exist_ok=True)

    labels_dataframe = pd.DataFrame(rows)
    scores_dataframe = pd.DataFrame(score_rows)

    labels_dataframe.to_csv(output_csv, index=False)
    scores_dataframe.to_csv(output_scores_csv, index=False)

    print(f"\nCSV binário criado em: {output_csv}")
    print(f"CSV de scores criado em: {output_scores_csv}")
    print(f"Total de frames: {len(rows)}")
    print(f"Total de labels: {len(SPECTRA_LABELS)}")

    if global_threshold is not None:
        print(f"Threshold global usado: {global_threshold}")
    else:
        print("Thresholds por grupo usados:")
        for group_name, threshold in DEFAULT_GROUP_THRESHOLDS.items():
            print(f"- {group_name}: {threshold}")

    print(f"Uso de top_k por grupo: {use_top_k}")


def main():
    parser = argparse.ArgumentParser(
        description="Baixa vídeos da Pixabay, extrai frames e cria labels automáticas binárias com CLIP."
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
            "woman dress",
            "person glasses",
            "people conversation",
            "office work",
            "restaurant",
            "school classroom",
            "beach",
            "park",
        ],
        help="Termos de busca para vídeos."
    )

    parser.add_argument(
        "--videos-per-query",
        type=int,
        default=3,
        help="Quantidade de vídeos por termo de busca."
    )

    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=8,
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
        help="CSV binário de saída com labels 0/1."
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

    parser.add_argument(
        "--global-threshold",
        type=float,
        default=None,
        help=(
            "Threshold global para converter score em 0/1. "
            "Exemplo: 0.5. Se não passar, usa thresholds por grupo."
        )
    )
    
    parser.add_argument(
        "--margin-scale",
        type=float,
        default=50.0,
        help="Escala aplicada na margem entre label e none. Valores maiores deixam a decisão mais extrema."
    )

    parser.add_argument(
        "--disable-top-k",
        action="store_true",
        help="Desativa o limite de top_k por grupo e aplica threshold em todas as labels."
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
        global_threshold=args.global_threshold,
        margin_scale=args.margin_scale,
        use_top_k=not args.disable_top_k,
    )


if __name__ == "__main__":
    main()