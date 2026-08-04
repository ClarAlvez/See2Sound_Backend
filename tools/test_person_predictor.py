from ai.spectra.inference.predictor import SpectraPredictor


def main():
    predictor = SpectraPredictor(
        model_path="data/models/spectra_person/person_net_best.pt",
        threshold=0.4,
        top_k=10,
        task_name="person",
    )

    image_path = (
        "data/external/market1501/"
        "Market-1501-v15.09.15/"
        "bounding_box_train/"
        "0002_c1s1_000451_03.jpg"
    )

    result = predictor.predict_frame(
        image_path=image_path,
        group_by_category=True,
    )

    print("\nImagem:", image_path)

    print("\nPredições:")
    for prediction in result["predictions"]:
        print(f"{prediction['label']}: {prediction['score']:.3f}")

    print("\nLabels finais:")
    print(result.get("labels", []))


if __name__ == "__main__":
    main()