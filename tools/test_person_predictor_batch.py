from pathlib import Path

from ai.spectra.inference.predictor import SpectraPredictor


def main():
    predictor = SpectraPredictor(
        model_path="data/models/spectra_person/person_net_best.pt",
        threshold=0.4,
        top_k=8,
        task_name="person",
    )

    images_dir = Path(
        "data/external/market1501/"
        "Market-1501-v15.09.15/"
        "bounding_box_test"
    )

    image_paths = sorted(images_dir.glob("*.jpg"))[:20]

    for image_path in image_paths:
        result = predictor.predict_frame(
            image_path=str(image_path),
            group_by_category=True,
        )

        print("\n" + "=" * 80)
        print("Imagem:", image_path.name)

        labels = result.get("labels", [])
        print("Labels:", labels)

        print("Top predições:")
        for prediction in result["predictions"][:8]:
            print(f"  {prediction['label']}: {prediction['score']:.3f}")


if __name__ == "__main__":
    main()