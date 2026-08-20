from typing import Dict, List

from ai.narrative.data_models import SceneContext


class SceneContextBuilder:
    """
    Organiza labels cruas da Spectra em categorias narrativas.

    Isso ajuda o modelo local a gerar frases mais fiéis
    e menos aleatórias.
    """

    def __init__(self):
        # ---------------------------------------------------------
        # Sujeitos
        # ---------------------------------------------------------

        self.subject_labels = {
            "person",
            "man",
            "woman",
            "child",
            "boy",
            "girl",
            "people",
            "crowd",
            "dog",
            "cat",
            "animal",
        }

        # ---------------------------------------------------------
        # Ações
        # ---------------------------------------------------------

        self.action_labels = {
            "running",
            "walking",
            "standing",
            "sitting",
            "talking",
            "looking",
            "smiling",
            "crying",
            "driving",
            "jumping",
            "fighting",
            "holding",
            "eating",
            "drinking",
            "playing",
            "reading",
            "writing",
            "sleeping",
            "dancing",
            "swimming",

            # Movimento genérico
            "moving",
        }

        # ---------------------------------------------------------
        # Ambientes
        # ---------------------------------------------------------

        self.environment_labels = {
            "street",
            "road",
            "city",
            "room",
            "house",
            "park",
            "forest",
            "beach",
            "school",
            "hospital",
            "restaurant",
            "store",
            "office",
            "kitchen",
            "bedroom",
            "bathroom",
            "car",
            "bus",
            "train",
            "building",
            "sidewalk",

            # Novos ambientes usados pela Spectra
            "field",
            "outdoor",
            "indoor",
        }

        # ---------------------------------------------------------
        # Tempo
        # ---------------------------------------------------------

        self.time_labels = {
            "day",
            "night",
            "morning",
            "afternoon",
            "evening",
            "sunset",
            "sunrise",
        }

        # ---------------------------------------------------------
        # Atributos
        # ---------------------------------------------------------

        self.attribute_labels = {
            # Ambiente
            "dark",
            "bright",
            "empty",
            "crowded",

            # Cores genéricas
            "red",
            "blue",
            "green",
            "black",
            "white",

            # Tamanho / estado
            "large",
            "small",
            "old",
            "new",
            "open",
            "closed",

            # Roupas
            "black_clothes",
            "white_clothes",
            "red_clothes",
            "blue_clothes",
            "green_clothes",
            "yellow_clothes",

            # Aparência
            "glasses",
            "short_hair",
            "long_hair",

            # Movimento
            "fast_motion",
            "slow_motion",
        }

        # ---------------------------------------------------------
        # Objetos
        # ---------------------------------------------------------

        self.object_labels = {
            "car",
            "bus",
            "bike",
            "bicycle",
            "motorcycle",
            "phone",
            "book",
            "table",
            "chair",
            "door",
            "window",
            "bag",
            "ball",
            "computer",
            "laptop",
            "bottle",
            "cup",
            "food",
            "tree",
            "traffic light",
        }

    def build(self, labels: List[str]) -> SceneContext:
        cleaned_labels = self._clean_labels(labels)

        context = SceneContext(
            raw_labels=cleaned_labels,
        )

        for label in cleaned_labels:
            categorized = False

            # Sujeitos
            if label in self.subject_labels:
                context.subjects.append(label)
                categorized = True

            # Ações
            if label in self.action_labels:
                context.actions.append(label)
                categorized = True

            # Ambiente
            if label in self.environment_labels:
                context.environment.append(label)
                categorized = True

            # Tempo
            if label in self.time_labels:
                context.time.append(label)
                categorized = True

            # Atributos
            if label in self.attribute_labels:
                context.attributes.append(label)
                categorized = True

            # Objetos
            if (
                label in self.object_labels
                and label not in context.environment
            ):
                context.objects.append(label)
                categorized = True

            # Tudo que não foi reconhecido
            if not categorized:
                context.unknown.append(label)

        return context

    def to_prompt_dict(
        self,
        scene_context: SceneContext,
    ) -> Dict[str, List[str]]:
        return {
            "raw_labels": scene_context.raw_labels,
            "subjects": scene_context.subjects,
            "actions": scene_context.actions,
            "objects": scene_context.objects,
            "environment": scene_context.environment,
            "time": scene_context.time,
            "attributes": scene_context.attributes,
            "unknown": scene_context.unknown,
        }

    def _clean_labels(
        self,
        labels: List[str],
    ) -> List[str]:
        cleaned = []

        for label in labels:
            if not isinstance(label, str):
                continue

            clean_label = label.strip().lower()

            if clean_label and clean_label not in cleaned:
                cleaned.append(clean_label)

        return cleaned