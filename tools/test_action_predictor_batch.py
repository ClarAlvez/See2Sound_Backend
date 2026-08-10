import argparse
import json
from pathlib import Path

from ai.spectra.Actions.inference import ActionPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Testa o modelo Actions em várias imagens."
    )

    parser.add_argument(
        "--images-dir",
        default="data/external/actions/human_action_recognition",
    )

    parser.add_argument(
        "--model-path",
        default="data/models/Actions/action_net_best.pt",
    )

    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    image_paths = []
    images_dir = Path(args.images_dir)

    for extension in [".jpg", ".jpeg", ".png", ".webp"]:
        image_paths.extend(images_dir.rglob(f"*{extension}"))

    image_paths = sorted(image_paths)[:args.limit]

    predictor = ActionPredictor(
        model_path=args.model_path,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    for image_path in image_paths:
        result = predictor.predict_frame(
            image_path=str(image_path),
            group_by_category=True,
        )

        print("\nImagem:", image_path)
        print(json.dumps(result["predictions"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()