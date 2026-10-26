import argparse
import json
import tempfile
from pathlib import Path





def json_default(value):
    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "tolist"):
        return value.tolist()

    raise TypeError(
        f"Tipo não serializável: {type(value)}"
    )


def save_json(
    data,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=json_default,
        )


def reset_directory(
    directory,
):
    """Create a fresh directory; refuse to overwrite an existing one."""
    directory = Path(
        directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )


def collect_frame_paths(
    frames_result,
):
    frames_output_dir = (
        frames_result.get(
            "frames_output_dir"
        )
    )

    if not frames_output_dir:
        raise ValueError(
            "extract_frames não retornou "
            "'frames_output_dir'."
        )

    frames_dir = Path(
        frames_output_dir
    )

    frame_paths = sorted(
        list(
            frames_dir.glob(
                "*.jpg"
            )
        )
        + list(
            frames_dir.glob(
                "*.jpeg"
            )
        )
        + list(
            frames_dir.glob(
                "*.png"
            )
        )
    )

    if not frame_paths:
        raise ValueError(
            "Nenhum frame foi extraído de "
            f"{frames_dir}"
        )

    return frame_paths


def create_person_predictor(
    model_path,
):
    from ai.spectra.predictor import SpectraPredictor

    if not model_path:
        return None

    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "Modelo Person não encontrado: "
            f"{model_path}"
        )

    return SpectraPredictor(
        model_path=str(
            model_path
        ),
        task_name="person",
        threshold=0.5,
        top_k=None,
    )


def create_object_analyzer(
    args,
):
    from ai.spectra.Object.object_analyzer import ObjectAnalyzer

    if not args.object_model:
        return None

    model_path = Path(
        args.object_model
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "Modelo Object não encontrado: "
            f"{model_path}"
        )

    return ObjectAnalyzer(
        object_model_path=str(
            model_path
        ),

        object_threshold=(
            args.object_threshold
        ),

        object_top_k=(
            args.object_top_k
        ),

        cropper_model_name=(
            args.object_cropper_model
        ),

        cropper_confidence_threshold=(
            args.object_cropper_threshold
        ),

        max_objects=(
            args.max_objects
        ),

        use_full_frame=False,
    )


def predict_person_attributes(
    predictor,
    crop_path,
):
    if predictor is None:
        return {
            "person": 1.0,
        }

    result = predictor.predict_frame(
        image_path=str(
            crop_path
        ),
        threshold=0.0,
        top_k=None,
        group_by_category=False,
    )

    attributes = {}

    for prediction in result.get(
        "predictions",
        [],
    ):
        label = prediction.get(
            "label"
        )

        score = prediction.get(
            "score"
        )

        if (
            label is None
            or score is None
        ):
            continue

        attributes[
            str(label)
        ] = float(
            score
        )

    if "person" not in attributes:
        attributes[
            "person"
        ] = 1.0

    return attributes


def analyze_people(
    frame_path,
    frame_id,
    person_cropper,
    person_predictor,
    people_output_dir,
    max_people,
):
    frame_output_dir = (
        people_output_dir
        / f"frame_{frame_id:06d}"
    )

    frame_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop_results = (
        person_cropper.crop_people(
            image_path=str(
                frame_path
            ),

            output_dir=str(
                frame_output_dir
            ),

            max_people=(
                max_people
            ),
        )
    )

    people = []

    for index, crop in enumerate(
        crop_results
    ):
        crop_path = crop.get(
            "crop_path"
        )

        if not crop_path:
            continue

        bbox = (
            crop.get(
                "detection_bbox"
            )
            or crop.get(
                "bbox"
            )
        )

        if (
            bbox is None
            or len(bbox) != 4
        ):
            continue

        attributes = (
            predict_person_attributes(
                predictor=(
                    person_predictor
                ),

                crop_path=(
                    crop_path
                ),
            )
        )

        people.append(
            {
                "detection_id": (
                    f"frame_{frame_id:06d}_"
                    f"person_{index:02d}"
                ),

                "bbox": [
                    float(value)
                    for value in bbox
                ],

                "crop_bbox": (
                    crop.get(
                        "bbox"
                    )
                ),

                "crop_path": str(
                    crop_path
                ),

                "detector_confidence": (
                    float(
                        crop.get(
                            "detector_confidence",
                            0.0,
                        )
                    )
                ),

                "attributes": (
                    attributes
                ),
            }
        )

    return people


def analyze_objects(
    frame_path,
    frame_id,
    object_analyzer,
    objects_output_dir,
    threshold,
    top_k,
):
    if object_analyzer is None:
        return None

    frame_output_dir = (
        objects_output_dir
        / f"frame_{frame_id:06d}"
    )

    frame_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return object_analyzer.analyze_frame(
        image_path=str(
            frame_path
        ),

        crops_output_dir=str(
            frame_output_dir
        ),

        threshold=threshold,

        top_k=top_k,
    )


def build_correlation_frames(
    frame_paths,
    args,
    person_cropper,
    person_predictor,
    object_analyzer,
    people_output_dir,
    objects_output_dir,
):
    correlation_frames = []

    people_debug = []
    object_debug = []

    total_frames = len(
        frame_paths
    )

    for frame_id, frame_path in enumerate(
        frame_paths
    ):
        timestamp = (
            frame_id
            * args.interval
        )

        print()
        print(
            f"[{frame_id + 1}/{total_frames}] "
            f"{frame_path.name}"
        )

        print(
            "  Analisando pessoas..."
        )

        people = analyze_people(
            frame_path=frame_path,

            frame_id=frame_id,

            person_cropper=(
                person_cropper
            ),

            person_predictor=(
                person_predictor
            ),

            people_output_dir=(
                people_output_dir
            ),

            max_people=(
                args.max_people
            ),
        )

        print(
            "  Pessoas detectadas:",
            len(people),
        )

        print(
            "  Analisando objetos..."
        )

        object_result = (
            analyze_objects(
                frame_path=frame_path,

                frame_id=frame_id,

                object_analyzer=(
                    object_analyzer
                ),

                objects_output_dir=(
                    objects_output_dir
                ),

                threshold=(
                    args.object_threshold
                ),

                top_k=(
                    args.object_top_k
                ),
            )
        )

        object_crop_count = 0

        if object_result:
            object_crop_count = len(
                object_result.get(
                    "crop_results",
                    object_result.get(
                        "raw_crop_results",
                        [],
                    ),
                )
            )

        print(
            "  Crops de objetos:",
            object_crop_count,
        )

        people_debug.append(
            {
                "frame_id": frame_id,

                "timestamp": (
                    timestamp
                ),

                "frame_path": str(
                    frame_path
                ),

                "people": (
                    people
                ),
            }
        )

        object_debug.append(
            {
                "frame_id": frame_id,

                "timestamp": (
                    timestamp
                ),

                "frame_path": str(
                    frame_path
                ),

                "result": (
                    object_result
                ),
            }
        )

        correlation_frames.append(
            {
                "frame_id": frame_id,

                "timestamp": (
                    timestamp
                ),

                "scene_id": None,

                "frame_path": str(
                    frame_path
                ),

                "people": (
                    people
                ),

                "raw": {
                    "object": (
                        object_result
                    ),
                },
            }
        )

    return (
        correlation_frames,
        people_debug,
        object_debug,
    )


def write_summary(
    result,
    output_path,
):
    lines = []

    lines.append(
        "SPECTRA CORRELATION v0.3"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Frames: "
        f"{result.get('frame_count', 0)}"
    )

    lines.append(
        f"Person tracks: "
        f"{result.get('track_count', 0)}"
    )

    lines.append(
        f"Person entities: "
        f"{result.get('entity_count', 0)}"
    )

    lines.append(
        f"Object tracks: "
        f"{result.get('object_track_count', 0)}"
    )

    lines.append(
        f"Object entities: "
        f"{result.get('object_count', 0)}"
    )

    relations = result.get(
        "person_object_relations",
        {},
    )

    relation_count = sum(
        len(items)
        for items
        in relations.values()
    )

    lines.append(
        f"Person-object relations: "
        f"{relation_count}"
    )

    lines.append("")
    lines.append(
        "ENTIDADES"
    )

    lines.append(
        "-" * 70
    )

    entities = result.get(
        "entities",
        {},
    )

    for entity_id, entity in (
        entities.items()
    ):
        lines.append(
            f"{entity_id}"
        )

        lines.append(
            "  tracks: "
            + ", ".join(
                entity.get(
                    "track_ids",
                    [],
                )
            )
        )

        lines.append(
            "  observacoes: "
            f"{entity.get('observation_count', 0)}"
        )

        lines.append(
            "  frames: "
            f"{entity.get('first_frame')} "
            f"-> "
            f"{entity.get('last_frame')}"
        )

        entity_relations = (
            relations.get(
                entity_id,
                [],
            )
        )

        if entity_relations:
            lines.append(
                "  objetos associados:"
            )

            for relation in (
                entity_relations
            ):
                persistent = (
                    relation.get(
                        "persistent",
                        False,
                    )
                )

                lines.append(
                    "    - "
                    f"{relation.get('relation')} "
                    f"{relation.get('object_label')} "
                    f"| obs="
                    f"{relation.get('observation_count')} "
                    f"| score="
                    f"{relation.get('best_score')} "
                    f"| persistent="
                    f"{persistent}"
                )

        lines.append("")

    lines.append(
        "OBJETOS"
    )

    lines.append(
        "-" * 70
    )

    objects = result.get(
        "objects",
        {},
    )

    for object_id, object_data in (
        objects.items()
    ):
        lines.append(
            f"{object_id}: "
            f"{object_data.get('label')}"
        )

        lines.append(
            "  tracks: "
            + ", ".join(
                object_data.get(
                    "track_ids",
                    [],
                )
            )
        )

        lines.append(
            "  observacoes: "
            f"{object_data.get('observation_count', 0)}"
        )

        lines.append("")

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(
                lines
            )
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Teste isolado da Spectra "
            "Correlation v0.3 com "
            "Person e Object."
        )
    )

    parser.add_argument(
        "video",
        help=(
            "Caminho para o vídeo "
            "de teste."
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help=(
            "Intervalo entre frames "
            "em segundos."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/output/"
            "correlation_test_v03"
        ),
    )

    parser.add_argument(
        "--person-model",
        default=(
            "data/models/"
            "spectra_person/"
            "person_net_best.pt"
        ),
    )

    parser.add_argument(
        "--person-cropper-model",
        default="yolov8n.pt",
    )

    parser.add_argument(
        "--person-cropper-threshold",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--max-people",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--object-model",
        default=(
            "data/models/"
            "spectra_object/"
            "object_net_best.pt"
        ),
    )

    parser.add_argument(
        "--object-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--object-top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--object-cropper-model",
        default="yolov8n.pt",
    )

    parser.add_argument(
        "--object-cropper-threshold",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--max-objects",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--identity-backbone",
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
        ],
    )

    return parser


def main():
    from ai.spectra.Person.person_cropper import PersonCropper
    from ai.spectra.Correlation.config import CorrelationConfig
    from ai.spectra.Correlation.engine import CorrelationEngine
    from pipeline.video.frames import extract_frames

    parser = build_parser()
    args = parser.parse_args()

    video_path = Path(
        args.video
    )

    if not video_path.exists():
        raise FileNotFoundError(
            "Vídeo não encontrado: "
            f"{video_path}"
        )

    output_parent = Path(args.output_dir)
    output_parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="run_", dir=output_parent))

    frames_output_dir = (
        output_dir
        / "frames"
    )

    people_output_dir = (
        output_dir
        / "people"
    )

    objects_output_dir = (
        output_dir
        / "objects"
    )

    reset_directory(
        frames_output_dir
    )

    reset_directory(
        people_output_dir
    )

    reset_directory(
        objects_output_dir
    )

    print(
        "=" * 80
    )

    print(
        "SPECTRA CORRELATION v0.3"
    )

    print(
        "=" * 80
    )

    print(
        "Vídeo:",
        video_path,
    )

    print(
        "Intervalo:",
        args.interval,
    )

    print()
    print(
        "[1/5] Extraindo frames..."
    )

    frames_result = extract_frames(
        video_path=str(
            video_path
        ),

        output_dir=str(
            frames_output_dir
        ),

        interval_seconds=(
            args.interval
        ),
    )

    frame_paths = (
        collect_frame_paths(
            frames_result
        )
    )

    print(
        "Frames extraídos:",
        len(frame_paths),
    )

    print()
    print(
        "[2/5] Carregando modelos..."
    )

    person_cropper = PersonCropper(
        model_name=(
            args.person_cropper_model
        ),

        confidence_threshold=(
            args.person_cropper_threshold
        ),
    )

    person_predictor = (
        create_person_predictor(
            args.person_model
        )
    )

    object_analyzer = (
        create_object_analyzer(
            args
        )
    )

    print(
        "PersonNet:",
        args.person_model,
    )

    print(
        "ObjectNet:",
        args.object_model,
    )

    print()
    print(
        "[3/5] Analisando frames..."
    )

    (
        correlation_frames,
        people_debug,
        object_debug,
    ) = build_correlation_frames(
        frame_paths=frame_paths,

        args=args,

        person_cropper=(
            person_cropper
        ),

        person_predictor=(
            person_predictor
        ),

        object_analyzer=(
            object_analyzer
        ),

        people_output_dir=(
            people_output_dir
        ),

        objects_output_dir=(
            objects_output_dir
        ),
    )

    people_debug_path = (
        output_dir
        / "people_detections.json"
    )

    object_debug_path = (
        output_dir
        / "object_detections.json"
    )

    input_debug_path = (
        output_dir
        / "correlation_input.json"
    )

    save_json(
        people_debug,
        people_debug_path,
    )

    save_json(
        object_debug,
        object_debug_path,
    )

    save_json(
        correlation_frames,
        input_debug_path,
    )

    print()
    print(
        "[4/5] Executando Correlation..."
    )

    config = CorrelationConfig(
        identity_encoder_type=(
            "resnet"
        ),

        identity_backbone=(
            args.identity_backbone
        ),
    )

    engine = CorrelationEngine(
        config=config
    )

    result = engine.process_video(
        correlation_frames
    )

    correlation_path = (
        output_dir
        / "correlation.json"
    )

    summary_path = (
        output_dir
        / "correlation_summary.txt"
    )

    save_json(
        result,
        correlation_path,
    )

    write_summary(
        result,
        summary_path,
    )

    print()
    print(
        "[5/5] Finalizado."
    )

    relations = result.get(
        "person_object_relations",
        {},
    )

    relation_count = sum(
        len(items)
        for items
        in relations.values()
    )

    print()
    print(
        "=" * 80
    )

    print(
        "RESULTADO"
    )

    print(
        "=" * 80
    )

    print(
        "Frames:",
        result.get(
            "frame_count",
            0,
        ),
    )

    print(
        "Person tracks:",
        result.get(
            "track_count",
            0,
        ),
    )

    print(
        "Person entities:",
        result.get(
            "entity_count",
            0,
        ),
    )

    print(
        "Object tracks:",
        result.get(
            "object_track_count",
            0,
        ),
    )

    print(
        "Object entities:",
        result.get(
            "object_count",
            0,
        ),
    )

    print(
        "Person-object relations:",
        relation_count,
    )

    print()
    print(
        "People debug:",
        people_debug_path,
    )

    print(
        "Object debug:",
        object_debug_path,
    )

    print(
        "Correlation input:",
        input_debug_path,
    )

    print(
        "Correlation:",
        correlation_path,
    )

    print(
        "Resumo:",
        summary_path,
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()