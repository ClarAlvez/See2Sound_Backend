from pathlib import Path
import argparse
import json

import torch
from PIL import Image

from ai.spectra.Scene.labels import LABELS
from ai.spectra.Scene.model import SpectraSceneNet
from ai.spectra.data.transforms import get_test_transforms


class ScenePredictor:
    def __init__(
        self,
        model_path,
        threshold=0.5,
        top_k=None,
        device=None,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo de cena não encontrado: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.labels = checkpoint.get("labels", LABELS)
        self.config = checkpoint.get("config", {})

        self.image_size = self.config.get("image_size", 224)
        self.dropout_rate = self.config.get("dropout_rate", 0.3)
        self.backbone_name = self.config.get("backbone_name", "resnet18")
        self.freeze_backbone = self.config.get("freeze_backbone", False)

        self.transform = get_test_transforms(self.image_size)

        self.model = SpectraSceneNet(
            output_size=len(self.labels),
            image_size=self.image_size,
            dropout_rate=self.dropout_rate,
            backbone_name=self.backbone_name,
            pretrained=False,
            freeze_backbone=self.freeze_backbone,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_frame(
        self,
        image_path,
        threshold=None,
        top_k=None,
        group_by_category=False,
    ):
        image_path = Path(image_path)

        image = self.load_image(image_path)

        with torch.no_grad():
            logits = self.model(image)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu()

        predictions = self.select_predictions(
            probabilities=probabilities,
            threshold=threshold,
            top_k=top_k,
        )

        result = {
            "frame_path": str(image_path),
            "task_name": "scene",
            "threshold": self.threshold if threshold is None else threshold,
            "predictions": predictions,
        }

        if group_by_category:
            result["grouped_predictions"] = {
                "scene": predictions,
                "person": [],
                "object": [],
            }

        return result

    def predict_top_labels(
        self,
        image_path,
        top_k=10,
    ):
        result = self.predict_frame(
            image_path=image_path,
            threshold=0.0,
            top_k=top_k,
        )

        return {
            "frame_path": result["frame_path"],
            "task_name": "scene",
            "top_predictions": result["predictions"],
        }

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def select_predictions(
        self,
        probabilities,
        threshold=None,
        top_k=None,
    ):
        threshold = self.threshold if threshold is None else threshold
        top_k = self.top_k if top_k is None else top_k

        values = []

        for label, score in zip(self.labels, probabilities):
            score_value = float(score)

            if score_value < threshold:
                continue

            values.append(
                {
                    "label": label,
                    "score": round(score_value, 4),
                }
            )

        values.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        values = self.apply_exclusive_groups(values)

        if top_k is not None:
            values = values[:top_k]

        return values

    def apply_exclusive_groups(self, predictions):
        exclusive_groups = [
            ["indoor", "outdoor"],
        ]

        predictions_by_label = {
            item["label"]: item
            for item in predictions
        }

        labels_to_remove = set()

        for group in exclusive_groups:
            active_labels = [
                label
                for label in group
                if label in predictions_by_label
            ]

            if len(active_labels) <= 1:
                continue

            best_label = max(
                active_labels,
                key=lambda label: predictions_by_label[label]["score"],
            )

            for label in active_labels:
                if label != best_label:
                    labels_to_remove.add(label)

        filtered_predictions = [
            item
            for item in predictions
            if item["label"] not in labels_to_remove
        ]

        return filtered_predictions


def build_parser():
    parser = argparse.ArgumentParser(
        description="Executa inferência com a SpectraSceneNet."
    )

    parser.add_argument(
        "image_path",
        help="Caminho da imagem para prever.",
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Caminho para o checkpoint scene_net_best.pt.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Threshold mínimo para exibir uma label.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Quantidade máxima de labels exibidas.",
    )

    parser.add_argument(
        "--top-only",
        action="store_true",
        help="Ignora threshold e mostra apenas as top-k labels.",
    )

    parser.add_argument(
        "--group-by-category",
        action="store_true",
        help="Inclui grouped_predictions no resultado.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Mostra resultado completo em JSON.",
    )

    return parser


def run_inference(args):
    predictor = ScenePredictor(
        model_path=args.model_path,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    if args.top_only:
        result = predictor.predict_top_labels(
            image_path=args.image_path,
            top_k=args.top_k,
        )

        predictions = result["top_predictions"]

    else:
        result = predictor.predict_frame(
            image_path=args.image_path,
            threshold=args.threshold,
            top_k=args.top_k,
            group_by_category=args.group_by_category,
        )

        predictions = result["predictions"]

    return result, predictions


def print_result(args, predictions):
    print("=" * 80)
    print("SPECTRA SCENE NET - INFERÊNCIA")
    print("=" * 80)
    print("Imagem:", args.image_path)
    print("Modelo:", args.model_path)

    if args.top_only:
        print("Modo: top-only")
    else:
        print("Threshold:", args.threshold)

    print("\nPredições:")

    if not predictions:
        print("Nenhuma label passou pelo threshold.")
        return

    for item in predictions:
        print(f"{item['label']}: {item['score']:.4f}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    result, predictions = run_inference(args)

    if args.json:
        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False,
            )
        )

        return

    print_result(
        args=args,
        predictions=predictions,
    )


if __name__ == "__main__":
    main()