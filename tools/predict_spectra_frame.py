import argparse

from ai.spectra.inference.predictor import SpectraPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Analisa um único frame usando a Spectra."
    )

    parser.add_argument(
        "frame_path",
        help="Caminho do frame/imagem que será analisado."
    )

    parser.add_argument(
        "--model-path",
        default="data/models/spectra_scene/scene_net_best.pt",
        help="Caminho do modelo treinado da Spectra."
    )

    parser.add_argument(
        "--task-name",
        default=None,
        choices=["scene", "person", "object", "all"],
        help="Força a task do modelo. Se não passar, usa a task salva no checkpoint."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold mínimo para retornar uma label."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Quantidade máxima de labels retornadas."
    )

    parser.add_argument(
        "--top-only",
        action="store_true",
        help="Mostra as labels mais prováveis ignorando o threshold."
    )

    args = parser.parse_args()

    predictor = SpectraPredictor(
        model_path=args.model_path,
        threshold=args.threshold,
        top_k=args.top_k,
        task_name=args.task_name,
    )

    if args.top_only:
        result = predictor.predict_top_labels(
            image_path=args.frame_path,
            top_k=args.top_k,
        )

        print("\nFrame analisado:", result["frame_path"])
        print("Task:", result["task_name"])
        print("\nLabels mais prováveis:")

        for prediction in result["top_predictions"]:
            print("{}: {:.4f}".format(
                prediction["label"],
                prediction["score"],
            ))

    else:
        result = predictor.predict_frame(
            image_path=args.frame_path,
            threshold=args.threshold,
            top_k=args.top_k,
            group_by_category=True,
        )

        print("\nFrame analisado:", result["frame_path"])
        print("Task:", result["task_name"])
        print("Threshold:", result["threshold"])

        print("\nPredições:")

        for prediction in result["predictions"]:
            print("{}: {:.4f}".format(
                prediction["label"],
                prediction["score"],
            ))

        print("\nPredições por grupo:")

        for group_name, predictions in result["grouped_predictions"].items():
            if not predictions:
                continue

            print("\n[{}]".format(group_name))

            for prediction in predictions:
                print("- {}: {:.4f}".format(
                    prediction["label"],
                    prediction["score"],
                ))


if __name__ == "__main__":
    main()