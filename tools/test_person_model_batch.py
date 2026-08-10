from pathlib import Path

from ai.spectra.Person.inference import PersonPredictor


MODEL_PATH = "data/models/Person_v2_age/person_net_best.pt"

TEST_DIRS = [
    "data/datasets/manual_review/person_crops",
    "data/external/utkface/UTKFace",
]


def collect_images(directory, limit=20):
    directory = Path(directory)

    if not directory.exists():
        return []

    images = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend(directory.rglob(extension))

    return images[:limit]


def main():
    predictor = PersonPredictor(
        model_path=MODEL_PATH,
        threshold=0.3,
        top_k=10,
    )

    print("Modelo carregado:", MODEL_PATH)
    print("Labels:", predictor.labels)
    print()

    all_images = []

    for directory in TEST_DIRS:
        all_images.extend(collect_images(directory, limit=10))

    if not all_images:
        print("Nenhuma imagem encontrada para teste.")
        return

    for image_path in all_images:
        result = predictor.predict_frame(
            image_path=str(image_path),
            group_by_category=True,
        )

        predictions_text = ", ".join(
            f"{item['label']}={item['score']}"
            for item in result["predictions"]
        )

        print(image_path)
        print(predictions_text)
        print("-" * 80)


if __name__ == "__main__":
    main()