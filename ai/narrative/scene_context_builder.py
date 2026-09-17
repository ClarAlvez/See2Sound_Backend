from typing import Dict, List

from ai.narrative.data_models import SceneContext


class SceneContextBuilder:
    """Organiza labels cruas da Spectra em categorias narrativas."""

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
            "moving",
        }

        self.environment_labels = {
            "indoor",
            "outdoor",
            "room",
            "street",
            "road",
            "city",
            "desert",
            "beach",
            "ocean",
            "forest",
            "mountain",
            "park",
            "field",
            "school",
            "classroom",
            "kitchen",
            "bedroom",
            "living_room",
            "office",
            "office_room",
            "office_cubicles",
            "home_office",
            "conference_room",
            "restaurant",
            "restaurant_indoor",
            "fastfood_restaurant",
            "cafeteria",
            "dining_room",
            "restaurant_patio",
            "store",
            "hospital",
            "sports_field",
            "house",
            "bathroom",
            "building",
            "sidewalk",
        }

        self.time_labels = {
            "day",
            "night",
            "dawn_dusk",
            "morning",
            "afternoon",
            "evening",
            "sunset",
            "sunrise",
        }

        self.attribute_labels = {
            "dark",
            "bright",
            "low_light",
            "clear_weather",
            "sunny",
            "cloudy",
            "foggy",
            "rainy",
            "snowy",
            "empty",
            "crowded",
            "red",
            "blue",
            "green",
            "black",
            "white",
            "large",
            "small",
            "old",
            "new",
            "open",
            "closed",
            "black_clothes",
            "white_clothes",
            "red_clothes",
            "blue_clothes",
            "green_clothes",
            "yellow_clothes",
            "glasses",
            "short_hair",
            "long_hair",
            "fast_motion",
            "slow_motion",
        }

        self.object_labels = {
            "car",
            "bus",
            "train",
            "boat",
            "airplane",
            "truck",
            "bike",
            "bicycle",
            "motorcycle",
            "phone",
            "book",
            "document",
            "table",
            "chair",
            "sofa",
            "bed",
            "door",
            "window",
            "bag",
            "backpack",
            "handbag",
            "suitcase",
            "ball",
            "computer",
            "screen",
            "television",
            "keyboard",
            "mouse",
            "remote",
            "bottle",
            "cup",
            "bowl",
            "plate",
            "fork",
            "spoon",
            "knife",
            "food",
            "fruit",
            "toy",
            "umbrella",
            "kite",
            "skateboard",
            "surfboard",
            "sports_racket",
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
