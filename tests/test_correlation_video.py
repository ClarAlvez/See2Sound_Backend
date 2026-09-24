from pathlib import Path
import argparse
import json

from pipeline.video.frames import extract_frames

from ai.spectra.Correlation.engine import (
    CorrelationEngine,
)

from ai.spectra.Correlation.person_adapter import (
    CorrelationPersonAdapter,
    normalize_pipeline_frames,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Teste end-to-end da "
            "Correlation v0.1."
        )
    )

    parser.add_argument(
        "video_path",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/output/"
            "correlation_test"
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-people",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--person-model",
        default=None,
    )

    args = parser.parse_args()

    video_path = Path(
        args.video_path
    )

    output_dir = Path(
        args.output_dir
    )

    frames_dir = (
        output_dir
        / "frames"
    )

    people_dir = (
        output_dir
        / "people"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nExtraindo frames..."
    )

    frames_result = extract_frames(
        video_path=str(
            video_path
        ),
        output_dir=str(
            frames_dir
        ),
        interval_seconds=(
            args.interval
        ),
    )

    frames = normalize_pipeline_frames(
        frames_result,
        frame_interval_seconds=args.interval,
    )

    print(
        "Frames:",
        len(frames),
    )

    print(
        "\nDetectando pessoas..."
    )

    person_predictor = None

    if args.person_model:
        from ai.spectra.predictor import (
            SpectraPredictor,
        )

        person_predictor = SpectraPredictor(
            model_path=args.person_model,
            task_name="person",
            threshold=0.5,
            top_k=None,
        )

    person_adapter = (
        CorrelationPersonAdapter(
            person_predictor=(
                person_predictor
            ),

            max_people=(
                args.max_people
            ),
        )
    )

    correlation_frames = (
        person_adapter.process_frames(
            frames=frames,
            output_dir=str(
                people_dir
            ),
        )
    )

    detections_path = (
        output_dir
        / "people_detections.json"
    )

    with open(
        detections_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "frames": (
                    correlation_frames
                )
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "Detecções salvas:",
        detections_path,
    )

    print(
        "\nExecutando Correlation..."
    )

    correlation_engine = (
        CorrelationEngine()
    )

    result = (
        correlation_engine.process_video(
            correlation_frames
        )
    )

    correlation_path = (
        output_dir
        / "correlation.json"
    )

    with open(
        correlation_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CORRELATION v0.1 "
        "- TESTE FINALIZADO"
    )

    print(
        "=" * 80
    )

    print(
        "Frames:",
        result[
            "frame_count"
        ],
    )

    print(
        "Entidades:",
        result[
            "entity_count"
        ],
    )

    print(
        "Resultado:",
        correlation_path,
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()