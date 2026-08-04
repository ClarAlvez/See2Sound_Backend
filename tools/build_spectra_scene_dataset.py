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

from ai.spectra.Scene.labels import SPECTRA_SCENE_LABELS


PIXABAY_VIDEO_API_URL = "https://pixabay.com/api/videos/"


SCENE_LABEL_GROUPS = {
    "environment": [
        "indoor",
        "outdoor",
        "room",
        "street",
        "school",
        "classroom",
        "kitchen",
        "bedroom",
        "living_room",
        "office",
        "forest",
        "city",
        "sports_field",
        "beach",
        "park",
        "store",
        "restaurant",
        "hospital",
    ],
    "lighting": [
        "dark_place",
        "bright_place",
        "day",
        "night",
    ],
    "composition": [
        "close_up",
        "medium_shot",
        "wide_shot",
        "empty_scene",
        "one_person",
        "two_people",
        "group_of_people",
        "crowded_scene",
    ],
    "state": [
        "calm_scene",
        "action_scene",
        "conversation_scene",
        "movement_scene",
    ],
    "action_hint": [
        "walking",
        "running",
        "sitting",
        "standing",
        "working",
        "playing",
        "driving",
        "dancing",
        "eating",
        "drinking",
    ],
}


SCENE_LABEL_PROMPTS = {
    "indoor": "an indoor scene",
    "outdoor": "an outdoor scene",
    "room": "an interior room",
    "street": "a street scene",
    "school": "a school environment",
    "classroom": "a classroom",
    "kitchen": "a kitchen",
    "bedroom": "a bedroom",
    "living_room": "a living room",
    "office": "an office",
    "forest": "a forest",
    "city": "an urban city scene",
    "sports_field": "a sports field",
    "beach": "a beach",
    "park": "a park",
    "store": "a store or shop",
    "restaurant": "a restaurant",
    "hospital": "a hospital",

    "dark_place": "a dark scene",
    "bright_place": "a brightly lit scene",
    "day": "a daytime scene",
    "night": "a nighttime scene",

    "close_up": "a close up shot",
    "medium_shot": "a medium shot",
    "wide_shot": "a wide shot",
    "empty_scene": "a scene with no visible people",
    "one_person": "a scene with one person",
    "two_people": "a scene with two people",
    "group_of_people": "a scene with a group of people",
    "crowded_scene": "a crowded scene with many people",

    "calm_scene": "a calm scene",
    "action_scene": "an action scene",
    "conversation_scene": "a conversation scene",
    "movement_scene": "a scene with visible movement",

    "walking": "a person walking",
    "running": "a person running",
    "sitting": "a person sitting",
    "standing": "a person standing",
    "working": "a person working",
    "playing": "a person playing",
    "driving": "a person driving",
    "dancing": "a person dancing",
    "eating": "a person eating",
    "drinking": "a person drinking",
}


GROUP_NONE_PROMPTS = {
    "environment": "a generic scene with no specific environment",
    "lighting": "a scene with no clear lighting condition",
    "composition": "a generic camera shot",
    "state": "a neutral scene",
    "action_hint": "a scene with no clear human action",
}


DEFAULT_GROUP_THRESHOLDS = {
    "environment": 0.58,
    "lighting": 0.58,
    "composition": 0.58,
    "state": 0.60,
    "action_hint": 0.62,
}


DEFAULT_GROUP_TOP_K = {
    "environment": 3,
    "lighting": 2,
    "composition": 2,
    "state": 2,
    "action_hint": 2,
}


LABEL_SPECIFIC_THRESHOLDS = {
    # Ambientes mais específicos precisam ser mais rígidos.
    "hospital": 0.70,
    "restaurant": 0.66,
    "classroom": 0.66,
    "school": 0.66,
    "sports_field": 0.66,
    "bedroom": 0.65,
    "living_room": 0.65,
    "kitchen": 0.64,
    "office": 0.64,

    # Ações visuais costumam gerar falso positivo em frame parado.
    "driving": 0.68,
    "dancing": 0.68,
    "eating": 0.67,
    "drinking": 0.67,
    "working": 0.65,
    "playing": 0.65,
    "running": 0.64,
    "walking": 0.63,

    # Estados de cena são interpretativos.
    "conversation_scene": 0.66,
    "action_scene": 0.65,
    "movement_scene": 0.63,
    "calm_scene": 0.64,

    # Composição com pessoas costuma confundir.
    "crowded_scene": 0.66,
    "group_of_people": 0.64,
    "two_people": 0.63,
    "one_person": 0.62,
    "empty_scene": 0.66,
}


MUTUALLY_EXCLUSIVE_GROUPS = [
    ["indoor", "outdoor"],
    ["day", "night"],
    ["dark_place", "bright_place"],
    ["close_up", "medium_shot", "wide_shot"],
    ["empty_scene", "one_person", "two_people", "group_of_people", "crowded_scene"],
    ["calm_scene", "action_scene", "conversation_scene", "movement_scene"],
]


QUERY_BOOSTS = {
    "kitchen": ["kitchen", "indoor"],
    "office": ["office", "indoor", "working"],
    "classroom": ["classroom", "school", "indoor"],
    "school": ["school", "classroom"],
    "street": ["street", "outdoor", "city"],
    "city": ["city", "street", "outdoor"],
    "forest": ["forest", "outdoor"],
    "beach": ["beach", "outdoor"],
    "park": ["park", "outdoor"],
    "restaurant": ["restaurant", "indoor"],
    "running": ["running", "movement_scene", "action_scene"],
    "walking": ["walking", "movement_scene"],
    "conversation": ["conversation_scene", "one_person", "two_people", "group_of_people"],
    "talking": ["conversation_scene"],
    "sports": ["sports_field", "running", "action_scene", "movement_scene"],
    "night": ["night", "dark_place"],
}


QUERY_BLOCKS = {
    "forest": ["kitchen", "office", "classroom", "school", "bedroom", "living_room", "hospital", "restaurant", "store"],
    "beach": ["kitchen", "office", "classroom", "school", "bedroom", "living_room", "hospital"],
    "street": ["kitchen", "bedroom", "living_room", "classroom"],
    "city": ["kitchen", "bedroom", "living_room", "forest"],
    "kitchen": ["forest", "beach", "sports_field", "street"],
    "office": ["forest", "beach", "sports_field"],
    "classroom": ["forest", "beach", "sports_field"],
    "bedroom": ["forest", "beach", "sports_field", "street"],
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
    return SCENE_LABEL_PROMPTS.get(label, label.replace("_", " "))


class ClipSceneAutoLabeler:
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
        text_features_by_group = {}

        with torch.no_grad():
            for group_name, labels in SCENE_LABEL_GROUPS.items():
                label_prompts = [
                    get_prompt_for_label(label)
                    for label in labels
                ]

                none_prompt = GROUP_NONE_PROMPTS.get(
                    group_name,
                    "a generic scene"
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
    def label_image(self, image_path: Path, source_query: str = "") -> Tuple[Dict[str, int], Dict[str, float]]:
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
            for label in SPECTRA_SCENE_LABELS
        }

        score_labels = {
            label: 0.0
            for label in SPECTRA_SCENE_LABELS
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

                confidence = self._apply_query_score_adjustment(
                    label=label,
                    score=confidence,
                    source_query=source_query,
                )

                score_labels[label] = round(confidence, 4)
                scored_labels.append((label, confidence))

            scored_labels.sort(
                key=lambda item: item[1],
                reverse=True
            )

            top_k = self._get_top_k_for_group(group_name)

            if self.use_top_k:
                candidates = scored_labels[:top_k]
            else:
                candidates = scored_labels

            for label, confidence in candidates:
                threshold = self._get_threshold_for_label(label, group_name)

                if confidence > threshold:
                    binary_labels[label] = 1
                else:
                    binary_labels[label] = 0

        self._apply_consistency_rules(binary_labels, score_labels)
        self._apply_minimum_scene_rules(binary_labels, score_labels)

        return binary_labels, score_labels

    def _apply_query_score_adjustment(self, label: str, score: float, source_query: str) -> float:
        query = source_query.lower()

        adjusted_score = score

        for keyword, labels in QUERY_BOOSTS.items():
            if keyword in query and label in labels:
                adjusted_score += 0.05

        for keyword, labels in QUERY_BLOCKS.items():
            if keyword in query and label in labels:
                adjusted_score -= 0.10

        if adjusted_score < 0.0:
            adjusted_score = 0.0

        if adjusted_score > 1.0:
            adjusted_score = 1.0

        return adjusted_score

    def _get_threshold_for_label(self, label: str, group_name: str) -> float:
        if label in LABEL_SPECIFIC_THRESHOLDS:
            return LABEL_SPECIFIC_THRESHOLDS[label]

        if self.global_threshold is not None:
            return self.global_threshold

        return DEFAULT_GROUP_THRESHOLDS.get(group_name, 0.58)

    def _get_top_k_for_group(self, group_name: str) -> int:
        return DEFAULT_GROUP_TOP_K.get(group_name, 2)

    def _apply_consistency_rules(self, binary_labels: Dict[str, int], score_labels: Dict[str, float]) -> None:
        def keep_only_best(labels):
            active_labels = [
                label for label in labels
                if binary_labels.get(label, 0) == 1
            ]

            if len(active_labels) <= 1:
                return

            best_label = max(
                active_labels,
                key=lambda label: score_labels.get(label, 0.0)
            )

            for label in active_labels:
                binary_labels[label] = 1 if label == best_label else 0

        for group in MUTUALLY_EXCLUSIVE_GROUPS:
            keep_only_best(group)

        if binary_labels.get("empty_scene", 0) == 1:
            for label in [
                "one_person",
                "two_people",
                "group_of_people",
                "crowded_scene",
                "conversation_scene",
                "walking",
                "running",
                "sitting",
                "standing",
                "working",
                "playing",
                "driving",
                "dancing",
                "eating",
                "drinking",
            ]:
                if label in binary_labels:
                    binary_labels[label] = 0

        if binary_labels.get("outdoor", 0) == 1:
            for label in [
                "room",
                "kitchen",
                "bedroom",
                "living_room",
                "office",
                "classroom",
            ]:
                if label in binary_labels and score_labels.get(label, 0.0) < 0.72:
                    binary_labels[label] = 0

        if binary_labels.get("indoor", 0) == 1:
            for label in [
                "forest",
                "beach",
                "sports_field",
                "street",
                "park",
            ]:
                if label in binary_labels and score_labels.get(label, 0.0) < 0.72:
                    binary_labels[label] = 0

    def _apply_minimum_scene_rules(self, binary_labels: Dict[str, int], score_labels: Dict[str, float]) -> None:
        """
        Garante que cada frame tenha pelo menos alguns sinais de cena.

        Isso evita frames totalmente zerados, mas usa apenas os scores mais fortes.
        """
        if not any(binary_labels.values()):
            best_labels = sorted(
                score_labels.items(),
                key=lambda item: item[1],
                reverse=True
            )[:2]

            for label, score in best_labels:
                if score >= 0.52:
                    binary_labels[label] = 1


def build_auto_scene_dataset(
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

    labeler = ClipSceneAutoLabeler(
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
                print(f"Rotulando cena: {frame_path}")

                labels, scores = labeler.label_image(
                    image_path=frame_path,
                    source_query=query,
                )

                row = {
                    "frame_path": str(frame_path),
                    "source": "pixabay",
                    "source_query": query,
                    "source_video_id": video_id,
                    "source_page_url": hit.get("pageURL", ""),
                }

                score_row = dict(row)

                for label in SPECTRA_SCENE_LABELS:
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

    print(f"\nCSV de cenas criado em: {output_csv}")
    print(f"CSV de scores criado em: {output_scores_csv}")
    print(f"Total de frames: {len(rows)}")
    print(f"Total de labels de cena: {len(SPECTRA_SCENE_LABELS)}")

    if global_threshold is not None:
        print(f"Threshold global usado: {global_threshold}")
    else:
        print("Thresholds por grupo/label usados.")

    print(f"Uso de top_k por grupo: {use_top_k}")


def main():
    parser = argparse.ArgumentParser(
        description="Baixa vídeos da Pixabay, extrai frames e cria dataset automático específico para SpectraSceneNet."
    )

    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "person walking street",
            "person running sports field",
            "people conversation office",
            "person sitting room",
            "kitchen cooking",
            "restaurant people eating",
            "office work computer",
            "school classroom students",
            "living room sofa",
            "bedroom interior",
            "city street cars",
            "forest landscape",
            "beach people",
            "park walking",
            "night city street",
            "hospital hallway",
            "store shopping",
            "person driving car",
            "person dancing",
            "person playing sports",
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
        default="data/dataset_sources/auto_pixabay_scene",
        help="Pasta onde vídeos e frames serão salvos."
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/spectra_scene_labels.csv",
        help="CSV binário de saída com labels de cena."
    )

    parser.add_argument(
        "--output-scores-csv",
        default="data/datasets/spectra_scene_scores.csv",
        help="CSV de saída com scores de cena."
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
        help="Threshold global opcional. Se omitido, usa thresholds por grupo/label."
    )

    parser.add_argument(
        "--margin-scale",
        type=float,
        default=50.0,
        help="Escala aplicada na margem entre label e none."
    )

    parser.add_argument(
        "--disable-top-k",
        action="store_true",
        help="Desativa top_k por grupo e aplica threshold em todas as labels."
    )

    args = parser.parse_args()

    build_auto_scene_dataset(
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


# python3 -m tools.build_spectra_scene_dataset_auto \
#   --queries \
#   "person walking street" \
#   "person running sports field" \
#   "people conversation office" \
#   "person sitting room" \
#   "kitchen cooking" \
#   "restaurant people eating" \
#   "office work computer" \
#   "school classroom students" \
#   "living room sofa" \
#   "bedroom interior" \
#   "city street cars" \
#   "forest landscape" \
#   "beach people" \
#   "park walking" \
#   "night city street" \
#   "hospital hallway" \
#   "store shopping" \
#   "person driving car" \
#   "person dancing" \
#   "person playing sports" \
#   "person standing outdoor" \
#   "empty street" \
#   "empty room" \
#   "crowded city street" \
#   "wide shot landscape" \
#   "close up person" \
#   "bright outdoor scene" \
#   "dark indoor scene" \
#   "calm nature scene" \
#   --videos-per-query 7 \
#   --frames-per-video 10 \
#   --interval 2 \
#   --video-quality tiny \
#   --output-dir data/dataset_sources/auto_pixabay_scene \
#   --output-csv data/datasets/spectra_scene_labels.csv \
#   --output-scores-csv data/datasets/spectra_scene_scores.csv \
#   --margin-scale 50
