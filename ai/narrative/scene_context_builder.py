from typing import Dict, List

from ai.narrative.data_models import SceneContext


class SceneContextBuilder:
    """
    Organiza labels cruas da Spectra em categorias narrativas.

    Isso ajuda o modelo local a gerar frases mais fiéis e menos aleatórias.
    """

    def __init__(self):
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
        }

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
        }

        self.time_labels = {
            "day",
            "night",
            "morning",
            "afternoon",
            "evening",
            "sunset",
            "sunrise",
        }

        self.attribute_labels = {
            "dark",
            "bright",
            "empty",
            "crowded",
            "red",
            "blue",
            "green",
            "large",
            "small",
            "old",
            "new",
            "open",
            "closed",
        }

        # Objetos comuns que não necessariamente são ambiente nem sujeito.
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

        context = SceneContext(raw_labels=cleaned_labels)

        for label in cleaned_labels:
            categorized = False

            if label in self.subject_labels:
                context.subjects.append(label)
                categorized = True

            if label in self.action_labels:
                context.actions.append(label)
                categorized = True

            if label in self.environment_labels:
                context.environment.append(label)
                categorized = True

            if label in self.time_labels:
                context.time.append(label)
                categorized = True

            if label in self.attribute_labels:
                context.attributes.append(label)
                categorized = True

            if label in self.object_labels and label not in context.environment:
                context.objects.append(label)
                categorized = True

            if not categorized:
                context.unknown.append(label)

        return context

    def to_prompt_dict(self, scene_context: SceneContext) -> Dict[str, List[str]]:
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

    def _clean_labels(self, labels: List[str]) -> List[str]:
        cleaned = []

        for label in labels:
            if not isinstance(label, str):
                continue

            clean_label = label.strip().lower()

            if clean_label and clean_label not in cleaned:
                cleaned.append(clean_label)

        return cleaned
