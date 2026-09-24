from pathlib import Path
import argparse
import json
from typing import Dict, List

import torch

from ai.spectra.Correlation.encoders import (
    create_identity_encoder,
)
from ai.spectra.Correlation.config import (
    CorrelationConfig,
)
from ai.spectra.Correlation.entities import (
    PersonDetection,
)
from ai.spectra.Correlation.memory import (
    EntityMemory,
)
from ai.spectra.Correlation.tracker import (
    PersonTracker,
)

from ai.spectra.Correlation.reconciler import (
    EntityReconciler,
)


class CorrelationEngine:
    def __init__(
        self,
        config=None,
    ):
        self.config = (
            config
            or CorrelationConfig()
        )

        self.identity_encoder = (
            create_identity_encoder(
                self.config
            )
        )

        self.reset_state()

        self.memory = EntityMemory(
            config=self.config
        )

        self.tracker = PersonTracker(
            config=self.config
        )

        self.current_scene_id = None

    def process_video(
        self,
        frames,
    ):
        self.reset_state()

        ordered_frames = sorted(
            frames,
            key=lambda frame: float(
                frame.get(
                    "timestamp",
                    0.0,
                )
            ),
        )

        processed_frames = []

        for frame in ordered_frames:
            processed = (
                self.process_frame(
                    frame
                )
            )

            processed_frames.append(
                processed
            )

        pre_reconciliation_entity_count = (
            len(
                self.memory.entities
            )
        )

        reconciliation_result = (
            self.reconciler.reconcile(
                processed_frames
            )
        )

        final_entity_count = (
            len(
                self.memory.entities
            )
        )

        return {
            "task_name": (
                "correlation"
            ),

            "version": (
                "0.2.1"
            ),

            "identity_encoder": {
                "type": (
                    self.config.identity_encoder_type
                ),

                "name": (
                    self.identity_encoder.name
                ),

                "embedding_dimension": (
                    self.identity_encoder.embedding_dimension
                ),
            },

            "frame_count": len(
                processed_frames
            ),

            "track_count": (
                self.tracker.total_tracks_created
            ),

            "entity_count_before_reconciliation": (
                pre_reconciliation_entity_count
            ),

            "entity_count": (
                final_entity_count
            ),

            "reconciliation": (
                reconciliation_result
            ),

            "frames": (
                processed_frames
            ),

            "entities": (
                self.memory.to_dict()
            ),
        }

    def process_frame(
        self,
        frame,
    ):
        frame_id = int(
            frame.get(
                "frame_id",
                0,
            )
        )

        timestamp = float(
            frame.get(
                "timestamp",
                0.0,
            )
        )

        scene_id = frame.get(
            "scene_id"
        )

        # ============================================================
        # Mudança de cena termina tracks locais,
        # mas NÃO apaga EntityMemory.
        # ============================================================

        if (
            self.config.reset_tracks_on_scene_change
            and scene_id is not None
            and self.current_scene_id is not None
            and scene_id
            != self.current_scene_id
        ):
            self.tracker.clear()

        if scene_id is not None:
            self.current_scene_id = (
                scene_id
            )

        raw_people = frame.get(
            "people",
            frame.get(
                "persons",
                [],
            ),
        )

        detections = (
            self.build_detections(
                raw_people=raw_people,
                frame_id=frame_id,
                timestamp=timestamp,
                scene_id=scene_id,
            )
        )

        self.generate_missing_embeddings(
            detections
        )

        # ============================================================
        # 1. Primeiro: tracking local
        # ============================================================

        track_matches = (
            self.tracker.match(
                detections=detections,
                timestamp=timestamp,
            )
        )

        # Evita que duas pessoas diferentes no mesmo
        # frame sejam associadas à mesma entity.
        assigned_entities = set()

        results = []

        for (
            detection_index,
            detection,
        ) in enumerate(
            detections
        ):

            # ========================================================
            # CASO 1:
            # Detecção continua um track existente
            # ========================================================

            if (
                detection_index
                in track_matches
            ):
                match = (
                    track_matches[
                        detection_index
                    ]
                )

                track_id = (
                    match["track_id"]
                )

                entity_id = (
                    match["entity_id"]
                )

                tracking_score = float(
                    match["score"]
                )

                # A identidade é herdada do track.
                #
                # NÃO fazemos Re-ID aqui.
                identity_source = (
                    "track_inheritance"
                )

                identity_score = 1.0

                self.memory.update_entity(
                    entity_id=entity_id,
                    detection=detection,
                )

                self.tracker.update_track(
                    track_id=track_id,
                    detection=detection,
                )

            # ========================================================
            # CASO 2:
            # Não pertence a nenhum track ativo
            # ========================================================

            else:
                tracking_score = None

                # Agora sim perguntamos à memória:
                # "este NOVO TRACK pertence a alguma
                # entidade que já conhecemos?"
                (
                    entity_id,
                    reid_score,
                ) = (
                    self.memory.find_reidentification(
                        detection=detection,

                        excluded_entity_ids=(
                            assigned_entities
                        ),
                    )
                )

                if entity_id is None:
                    entity_id = (
                        self.memory.create_entity(
                            detection
                        )
                    )

                    identity_source = (
                        "new_entity"
                    )

                    identity_score = 1.0

                else:
                    identity_source = (
                        "reidentification"
                    )

                    identity_score = float(
                        reid_score
                    )

                    self.memory.update_entity(
                        entity_id=entity_id,
                        detection=detection,
                    )

                # Criamos o track SOMENTE depois
                # de resolver qual entity ele representa.
                track_id = (
                    self.tracker.create_track(
                        entity_id=entity_id,
                        detection=detection,
                    )
                )

                self.memory.register_track(
                    entity_id=entity_id,
                    track_id=track_id,
                )

            assigned_entities.add(
                entity_id
            )

            results.append(
                {
                    "detection_id": (
                        detection.detection_id
                    ),

                    "track_id": (
                        track_id
                    ),

                    "entity_id": (
                        entity_id
                    ),

                    "tracking_source": (
                        "existing_track"
                        if tracking_score
                        is not None
                        else "new_track"
                    ),

                    "tracking_score": (
                        round(
                            tracking_score,
                            4,
                        )
                        if tracking_score
                        is not None
                        else None
                    ),

                    "identity_source": (
                        identity_source
                    ),

                    "identity_score": (
                        round(
                            float(
                                identity_score
                            ),
                            4,
                        )
                    ),

                    # Mantém compatibilidade com
                    # versões anteriores.
                    "correlation_source": (
                        identity_source
                    ),

                    "correlation_score": (
                        round(
                            float(
                                identity_score
                            ),
                            4,
                        )
                    ),

                    "bbox": list(
                        detection.bbox
                    ),

                    "crop_path": (
                        detection.crop_path
                    ),

                    "attributes": (
                        detection.attributes
                    ),
                }
            )

        return {
            "frame_id": frame_id,

            "timestamp": timestamp,

            "scene_id": scene_id,

            "people": results,
        }

    def reset_state(self):
        self.memory = EntityMemory(
            config=self.config
        )

        self.tracker = PersonTracker(
            config=self.config
        )

        self.reconciler = EntityReconciler(
            memory=self.memory,
            config=self.config,
        )

        self.current_scene_id = None

    def build_detections(
        self,
        raw_people,
        frame_id,
        timestamp,
        scene_id,
    ):
        detections = []

        for index, person in enumerate(
            raw_people
        ):
            bbox = person.get(
                "bbox"
            )

            if bbox is None:
                continue

            if len(bbox) != 4:
                raise ValueError(
                    "bbox precisa possuir "
                    "4 valores: "
                    "[x1, y1, x2, y2]"
                )

            embedding = (
                person.get(
                    "embedding"
                )
            )

            if embedding is not None:
                embedding = torch.tensor(
                    embedding,
                    dtype=torch.float32,
                )

            detection = PersonDetection(
                detection_id=(
                    person.get(
                        "detection_id",
                        (
                            f"frame_"
                            f"{frame_id}_"
                            f"person_"
                            f"{index}"
                        ),
                    )
                ),

                frame_id=frame_id,

                timestamp=timestamp,

                bbox=tuple(
                    float(value)
                    for value in bbox
                ),

                crop_path=(
                    person.get(
                        "crop_path"
                    )
                ),

                attributes=(
                    person.get(
                        "attributes",
                        {},
                    )
                ),

                embedding=embedding,

                scene_id=scene_id,
            )

            detections.append(
                detection
            )

        return detections

    def generate_missing_embeddings(
        self,
        detections,
    ):
        pending_detections = [
            detection
            for detection in detections
            if (
                detection.embedding
                is None
                and detection.crop_path
            )
        ]

        if not pending_detections:
            return

        crop_paths = [
            detection.crop_path
            for detection
            in pending_detections
        ]

        embeddings = (
            self.identity_encoder.encode_batch(
                crop_paths
            )
        )

        for (
            detection,
            embedding,
        ) in zip(
            pending_detections,
            embeddings,
        ):
            detection.embedding = (
                embedding
            )


def resolve_crop_paths(
    frames,
    base_directory,
):
    base_directory = Path(
        base_directory
    )

    for frame in frames:
        people = frame.get(
            "people",
            frame.get(
                "persons",
                [],
            ),
        )

        for person in people:
            crop_path = person.get(
                "crop_path"
            )

            if not crop_path:
                continue

            path = Path(
                crop_path
            )

            if path.is_absolute():
                continue

            person[
                "crop_path"
            ] = str(
                (
                    base_directory
                    / path
                ).resolve()
            )


def load_input_json(
    input_path,
):
    input_path = Path(
        input_path
    )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if isinstance(
        data,
        list,
    ):
        frames = data

    elif isinstance(
        data,
        dict,
    ):
        frames = data.get(
            "frames",
            []
        )

    else:
        raise ValueError(
            "O JSON precisa ser uma lista "
            "de frames ou possuir "
            "uma chave 'frames'."
        )

    resolve_crop_paths(
        frames=frames,
        base_directory=(
            input_path.parent
        ),
    )

    return frames


def save_output_json(
    result,
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
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Executa a Spectra "
            "Correlation v0.1."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "JSON contendo os frames "
            "e detecções de pessoas."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Arquivo JSON onde a "
            "correlação será salva."
        ),
    )

    parser.add_argument(
        "--short-term-threshold",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.76,
    )

    parser.add_argument(
        "--max-track-seconds",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--keep-tracks-between-scenes",
        action="store_true",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = CorrelationConfig(
        short_term_min_score=(
            args.short_term_threshold
        ),

        reidentification_min_score=(
            args.reid_threshold
        ),

        short_term_max_seconds=(
            args.max_track_seconds
        ),

        reset_tracks_on_scene_change=(
            not args.keep_tracks_between_scenes
        ),
    )

    frames = load_input_json(
        args.input
    )

    engine = CorrelationEngine(
        config=config
    )

    result = engine.process_video(
        frames
    )

    save_output_json(
        result=result,
        output_path=args.output,
    )

    print(
        "=" * 80
    )

    print(
        "SPECTRA CORRELATION v0.1"
    )

    print(
        "=" * 80
    )

    print(
        "Frames processados:",
        result["frame_count"],
    )

    print(
        "Entidades encontradas:",
        result["entity_count"],
    )

    print(
        "Resultado:",
        args.output,
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()