from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torchvision import transforms

from ai.spectra.predictor import SpectraPredictor
from ai.spectra.Person.labels import LABELS


def main():
    model_path = "data/models/spectra_person_v3_hair/person_net_best.pt"
    csv_path = "data/datasets/spectra_person_v3_hair_labels.csv"

    predictor = SpectraPredictor(
        model_path=model_path,
        threshold=0.5,
        top_k=len(LABELS),
        task_name="person",
    )

    df = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    # Amostra para não demorar demais
    df = df.sample(
        n=min(1000, len(df)),
        random_state=42,
    ).reset_index(drop=True)

    y_true = []
    y_pred = []

    for index, row in df.iterrows():
        image_path = Path(row["frame_path"])

        if not image_path.exists():
            continue

        result = predictor.predict_frame(
            image_path=str(image_path),
            group_by_category=True,
        )

        prediction_scores = {
            prediction["label"]: prediction["score"]
            for prediction in result["predictions"]
        }

        true_row = []
        pred_row = []

        for label in LABELS:
            true_row.append(int(row.get(label, 0)))
            pred_row.append(1 if prediction_scores.get(label, 0.0) >= 0.5 else 0)

        y_true.append(true_row)
        y_pred.append(pred_row)

        if (index + 1) % 100 == 0:
            print(f"Avaliadas {index + 1} imagens...")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    report = pd.DataFrame(
        {
            "label": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    report = report.sort_values("f1", ascending=True)

    output_path = Path(
        "data/training_reports/spectra_person/person_v3_hair_per_label_report.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print("\nRelatório por label:")
    print(report)

    print("\nSalvo em:")
    print(output_path)


if __name__ == "__main__":
    main()
