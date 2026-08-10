from collections import Counter, defaultdict
from pathlib import Path

from ai.spectra.predictor import SpectraPredictor


def split_predictions_by_group(predictions):
    from ai.spectra.Object.labels import SPECTRA_OBJECT_LABELS
    from ai.spectra.Person.labels import LABELS
    from ai.spectra.Scene.labels import SPECTRA_SCENE_LABELS

    groups = {"scene": [], "person": [], "object": []}
    label_sets = {"scene": SPECTRA_SCENE_LABELS, "person": LABELS, "object": SPECTRA_OBJECT_LABELS}
    for prediction in predictions:
        for name, labels in label_sets.items():
            if prediction["label"] in labels:
                groups[name].append(prediction)
    return groups


class Spectra:
    """
    Interface principal da Spectra.

    Essa classe é a entrada oficial para usar a IA visual dentro do backend.

    Ela pode:
    - analisar um frame isolado
    - analisar vários frames
    - analisar frames agrupados por cena
    - gerar um resumo visual simples por cena

    A Spectra não gera audiodescrição final aqui.
    Ela gera uma análise visual estruturada para o módulo narrativo usar depois.
    """

    def __init__(
        self,
        model_path="data/models/spectra_scene/scene_net_best.pt",
        threshold=0.5,
        top_k=10,
    ):
        self.predictor = SpectraPredictor(
            model_path=model_path,
            threshold=threshold,
            top_k=top_k,
        )

    def analyze_frame(self, frame_path, timestamp=None, group_by_category=True):
        """
        Analisa um único frame.

        Retorno:
            {
                "frame_path": "...",
                "timestamp": 10.0,
                "predictions": [...],
                "grouped_predictions": {...}
            }
        """
        result = self.predictor.predict_frame(
            image_path=frame_path,
            group_by_category=group_by_category,
        )

        result["timestamp"] = timestamp

        return result

    def analyze_top_frame_labels(self, frame_path, top_k=10):
        """
        Retorna as labels mais prováveis de um frame.

        Útil para debug e avaliação rápida do modelo.
        """
        return self.predictor.predict_top_labels(
            image_path=frame_path,
            top_k=top_k,
        )

    def analyze_frames(self, frames):
        """
        Analisa vários frames.

        Espera uma lista no formato:

            [
                {
                    "frame_path": "data/output/frames/frame_001.jpg",
                    "timestamp": 1.0
                },
                {
                    "frame_path": "data/output/frames/frame_002.jpg",
                    "timestamp": 2.0
                }
            ]

        Retorna uma lista de análises.
        """
        analyses = []

        for frame in frames:
            frame_path = frame["frame_path"]
            timestamp = frame.get("timestamp")

            analysis = self.analyze_frame(
                frame_path=frame_path,
                timestamp=timestamp,
                group_by_category=True,
            )

            analyses.append(analysis)

        return analyses

    def analyze_scene(self, scene, frames):
        """
        Analisa uma cena específica.

        scene esperado:

            {
                "scene_id": 1,
                "start_time": 10.0,
                "end_time": 15.5
            }

        frames esperado:

            [
                {
                    "frame_path": "...",
                    "timestamp": 10.0
                }
            ]

        A função filtra os frames que pertencem ao intervalo da cena.
        """
        scene_id = scene.get("scene_id")
        start_time = scene["start_time"]
        end_time = scene["end_time"]

        scene_frames = self._filter_frames_by_time(
            frames=frames,
            start_time=start_time,
            end_time=end_time,
        )

        frame_analyses = self.analyze_frames(scene_frames)

        scene_summary = self._summarize_scene(
            frame_analyses=frame_analyses,
        )

        return {
            "scene_id": scene_id,
            "start_time": start_time,
            "end_time": end_time,
            "frames_analyzed_count": len(frame_analyses),
            "frame_analyses": frame_analyses,
            "scene_summary": scene_summary,
        }

    def analyze_scenes(self, scenes, frames):
        """
        Analisa várias cenas do vídeo.

        scenes esperado:

            [
                {
                    "scene_id": 1,
                    "start_time": 0.0,
                    "end_time": 5.0
                },
                {
                    "scene_id": 2,
                    "start_time": 5.0,
                    "end_time": 10.0
                }
            ]

        frames esperado:

            [
                {
                    "frame_path": "...",
                    "timestamp": 1.0
                }
            ]
        """
        scene_analyses = []

        for index, scene in enumerate(scenes):
            if "scene_id" not in scene:
                scene["scene_id"] = index + 1

            analysis = self.analyze_scene(
                scene=scene,
                frames=frames,
            )

            scene_analyses.append(analysis)

        return {
            "scenes_analyzed_count": len(scene_analyses),
            "scene_analyses": scene_analyses,
        }

    def _filter_frames_by_time(self, frames, start_time, end_time):
        """
        Filtra frames pelo timestamp.
        """
        selected_frames = []

        for frame in frames:
            timestamp = frame.get("timestamp")

            if timestamp is None:
                continue

            if start_time <= timestamp <= end_time:
                selected_frames.append(frame)

        return selected_frames

    def _summarize_scene(self, frame_analyses):
        """
        Gera um resumo simples da cena com base nas labels mais frequentes.

        Esse resumo ainda não é audiodescrição final.
        Ele serve como entrada para o módulo narrativo.
        """
        label_counter = Counter()
        label_score_sum = defaultdict(float)
        grouped_counter = defaultdict(Counter)

        for frame_analysis in frame_analyses:
            predictions = frame_analysis.get("predictions", [])

            for prediction in predictions:
                label = prediction["label"]
                score = prediction["score"]

                label_counter[label] += 1
                label_score_sum[label] += score

            grouped_predictions = frame_analysis.get("grouped_predictions")

            if grouped_predictions is None:
                grouped_predictions = split_predictions_by_group(predictions)

            for group_name, group_predictions in grouped_predictions.items():
                for prediction in group_predictions:
                    grouped_counter[group_name][prediction["label"]] += 1

        main_labels = self._get_main_labels(
            label_counter=label_counter,
            label_score_sum=label_score_sum,
            limit=10,
        )

        grouped_summary = {}

        for group_name, counter in grouped_counter.items():
            grouped_summary[group_name] = [
                label
                for label, count in counter.most_common(5)
            ]

        return {
            "main_labels": main_labels,
            "grouped_summary": grouped_summary,
        }

    def _get_main_labels(self, label_counter, label_score_sum, limit=10):
        """
        Ordena labels por frequência e score médio.
        """
        label_data = []

        for label, count in label_counter.items():
            average_score = label_score_sum[label] / count

            label_data.append({
                "label": label,
                "count": count,
                "average_score": round(average_score, 4),
            })

        label_data.sort(
            key=lambda item: (item["count"], item["average_score"]),
            reverse=True,
        )

        return label_data[:limit]
