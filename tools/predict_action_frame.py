import argparse
import json
from pathlib import Path

from ai.spectra.Actions.inference import ActionPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Testa o modelo Actions em uma imagem/frame."
    )

    parser.add_argument(
        "--image-path",
        required=True,
    )

    parser.add_argument(
        "--model-path",
        default="data/models/Actions/action_net_best.pt",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    predictor = ActionPredictor(
        model_path=args.model_path,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    result = predictor.predict_frame(
        image_path=args.image_path,
        group_by_category=True,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()