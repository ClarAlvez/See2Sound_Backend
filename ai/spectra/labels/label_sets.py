"""
Conjunto de labels da Spectra.

A Spectra utiliza classificação multilabel, ou seja, uma mesma imagem pode
possuir várias labels verdadeiras ao mesmo tempo.

Exemplo:
    Uma pessoa sentada em uma cozinha durante o dia pode ativar:
    - person
    - sitting
    - kitchen
    - indoor
    - day
    - medium_shot
"""

PERSON_LABELS = [
    "person",
    "face",
    "hand",
    "man",
    "woman",
    "child",
    "group_of_people",
]

COMMON_OBJECT_LABELS = [
    "book",
    "table",
    "chair",
    "sofa",
    "bed",
    "door",
    "window",
    "phone",
    "computer",
    "screen",
    "television",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "animal",
    "dog",
    "cat",
    "food",
    "cup",
    "bottle",
    "bag",
    "backpack",
    "weapon",
    "knife",
    "ball",
    "toy",
    "paper",
    "box",
]

SPECIFIC_OBJECT_LABELS = [
    "dice",
    "miniature",
    "board_game",
    "musical_instrument",
    "cap",
    "subtitles",
    "on_screen_text",
]

ACTION_LABELS = [
    "sitting",
    "standing",
    "walking",
    "running",
    "talking",
    "looking",
    "holding",
    "reading",
    "writing",
    "playing",
    "fighting",
    "falling",
    "jumping",
    "pointing",
    "smiling",
    "crying",
    "opening",
    "closing",
    "showing",
    "eating",
    "drinking",
    "driving",
    "dancing",
    "working",
]

SCENARIO_LABELS = [
    "indoor",
    "outdoor",
    "room",
    "street",
    "school",
    "classroom",
    "kitchen",
    "bedroom",
    "living_room",
    "office",
    "forest",
    "city",
    "sports_field",
    "beach",
    "park",
    "store",
    "restaurant",
    "hospital",
    "dark_place",
    "bright_place",
    "day",
    "night",
]

HAIR_LABELS = [
    "blonde_hair",
    "brown_hair",
    "black_hair",
    "red_hair",
    "gray_hair",
    "short_hair",
    "long_hair",
    "curly_hair",
    "straight_hair",
]

CLOTHING_LABELS = [
    "red_clothes",
    "blue_clothes",
    "black_clothes",
    "white_clothes",
    "green_clothes",
    "yellow_clothes",
    "dress",
    "shirt",
    "jacket",
    "hat",
    "glasses",
]

SKIN_TONE_LABELS = [
    "light_skin",
    "medium_skin",
    "dark_skin",
]

COMPOSITION_LABELS = [
    "close_up",
    "medium_shot",
    "wide_shot",
    "one_person",
    "two_people",
    "crowded_scene",
    "empty_scene",
]

VISUAL_STATE_LABELS = [
    "calm_scene",
    "action_scene",
    "conversation_scene",
    "movement_scene",
]

SPECTRA_LABEL_GROUPS = {
    "person": PERSON_LABELS,
    "common_objects": COMMON_OBJECT_LABELS,
    "specific_objects": SPECIFIC_OBJECT_LABELS,
    "actions": ACTION_LABELS,
    "scenarios": SCENARIO_LABELS,
    "composition": COMPOSITION_LABELS,
    "visual_state": VISUAL_STATE_LABELS,
    "hair": HAIR_LABELS,
    "clothing": CLOTHING_LABELS,
    "skin_tone": SKIN_TONE_LABELS,
}


SPECTRA_LABELS = (
    PERSON_LABELS
    + COMMON_OBJECT_LABELS
    + SPECIFIC_OBJECT_LABELS
    + ACTION_LABELS
    + SCENARIO_LABELS
    + COMPOSITION_LABELS
    + VISUAL_STATE_LABELS
    + HAIR_LABELS
    + CLOTHING_LABELS
    + SKIN_TONE_LABELS
)


def get_label_count():
    """
    Retorna a quantidade total de labels.
    """
    return len(SPECTRA_LABELS)


def get_label_index(label):
    """
    Retorna o índice de uma label dentro de SPECTRA_LABELS.
    """
    if label not in SPECTRA_LABELS:
        raise ValueError("Label não encontrada: {}".format(label))

    return SPECTRA_LABELS.index(label)


def get_labels_by_group(group_name):
    """
    Retorna as labels de um grupo específico.

    Exemplo:
        get_labels_by_group("actions")
    """
    if group_name not in SPECTRA_LABEL_GROUPS:
        raise ValueError("Grupo de labels não encontrado: {}".format(group_name))

    return SPECTRA_LABEL_GROUPS[group_name]


def split_predictions_by_group(predictions):
    """
    Separa uma lista de predições por grupo.

    Entrada:
        [
            {"label": "person", "score": 0.91},
            {"label": "running", "score": 0.74}
        ]

    Saída:
        {
            "person": [...],
            "actions": [...]
        }
    """
    grouped_predictions = {
        group_name: []
        for group_name in SPECTRA_LABEL_GROUPS.keys()
    }

    for prediction in predictions:
        label = prediction["label"]

        for group_name, labels in SPECTRA_LABEL_GROUPS.items():
            if label in labels:
                grouped_predictions[group_name].append(prediction)
                break

    return grouped_predictions