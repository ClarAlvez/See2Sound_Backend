import argparse
import json
from pathlib import Path

from ai.spectra.Actions.analyzer import PersonActionAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Detecta pessoas no frame e roda Actions nos crops."
    )

    parser.add_argument("--image-path", required=True)

    parser.add_argument(
        "--action-model-path",
        default="data/models/Actions/action_net_best.pt",
    )

    parser.add_argument(
        "--crops-output-dir",
        default="data/output/action_person_crops",
    )

    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-people", type=int, default=5)

    args = parser.parse_args()

    analyzer = PersonActionAnalyzer(
        action_model_path=args.action_model_path,
        action_threshold=args.threshold,
        action_top_k=args.top_k,
        max_people=args.max_people,
    )

    result = analyzer.analyze_frame(
        image_path=args.image_path,
        crops_output_dir=args.crops_output_dir,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()