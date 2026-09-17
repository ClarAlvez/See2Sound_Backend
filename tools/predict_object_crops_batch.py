import argparse
from pathlib import Path
from typing import List, Set

import pandas as pd

from ai.spectra.Object.labels import LABELS
from ai.spectra.Object.object_analyzer import ObjectAnalyzer


def get_expected_labels(row: pd.Series) -> Set[str]:
    expected = set()

    for label in LABELS:
        if label in row.index:
            try:
                value = int(row[label])
            except Exception:
                value = 0

            if value == 1:
                expected.add(label)

    return expected


def parse_predicted_labels(predictions: List[dict]) -> Set[str]:
    return {
        prediction["label"]
        for prediction in predictions
        if prediction.get("label")
    }


def calculate_counts(expected: Set[str], predicted: Set[str]):
    true_positive = len(expected & predicted)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)

    return true_positive, false_positive, false_negative


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser(
        description="Teste em lote do ObjectAnalyzer com cropper."
    )

    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--crops-output-dir", default="data/output/object_batch_crops")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cropper-model", default="yolov8n.pt")
    parser.add_argument("--cropper-confidence", type=float, default=0.25)
    parser.add_argument("--max-objects", type=int, default=20)
    parser.add_argument("--no-full-frame", action="store_true")

    args = parser.parse_args()

    input_csv = Path(args.csv)
    output_csv = Path(args.output_csv)
    crops_output_dir = Path(args.crops_output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV não encontrado: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    analyzer = ObjectAnalyzer(
        object_model_path=args.model_path,
        object_threshold=args.threshold,
        object_top_k=args.top_k,
        cropper_model_name=args.cropper_model,
        cropper_confidence_threshold=args.cropper_confidence,
        max_objects=args.max_objects,
        use_full_frame=not args.no_full_frame,
    )

    rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for index, row in df.iterrows():
        image_path = Path(str(row["frame_path"]))

        if not image_path.exists():
            continue

        expected_labels = get_expected_labels(row)

        image_crop_dir = crops_output_dir / f"sample_{index:05d}"

        result = analyzer.analyze_frame(
            image_path=str(image_path),
            crops_output_dir=str(image_crop_dir),
            threshold=args.threshold,
            top_k=args.top_k,
        )

        predicted_labels = parse_predicted_labels(result.get("predictions", []))

        tp, fp, fn = calculate_counts(expected_labels, predicted_labels)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)

        rows.append(
            {
                "image_path": str(image_path),
                "expected_labels": ";".join(sorted(expected_labels)),
                "predicted_labels": ";".join(sorted(predicted_labels)),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "crops_detected": result.get("crops_detected", 0),
            }
        )

        print(
            f"[{len(rows)}/{len(df)}] "
            f"TP={tp} FP={fp} FN={fn} "
            f"F1={f1:.4f} "
            f"{image_path.name}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    total_precision = safe_ratio(total_tp, total_tp + total_fp)
    total_recall = safe_ratio(total_tp, total_tp + total_fn)
    total_f1 = safe_ratio(
        2 * total_precision * total_recall,
        total_precision + total_recall,
    )

    print("\nResultado geral")
    print("TP:", total_tp)
    print("FP:", total_fp)
    print("FN:", total_fn)
    print("Precision:", round(total_precision, 4))
    print("Recall:", round(total_recall, 4))
    print("F1:", round(total_f1, 4))
    print("\nCSV salvo em:")
    print(output_csv)


if __name__ == "__main__":
    main()