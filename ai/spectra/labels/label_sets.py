"""
Label sets da Spectra.

A Spectra é dividida em submodelos:

- SpectraSceneNet:
    Analisa o frame inteiro.

- SpectraPersonNet:
    Analisa recortes de pessoas.

- SpectraObjectNet:
    Analisa objetos importantes.

- SpectraCorrelationNet:
    Relaciona cena, pessoa, objeto e fala ao longo do tempo.
"""


# ============================================================
# 1. Labels do modelo de cena
# ============================================================

SCENE_ENVIRONMENT_LABELS = [
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
]

SCENE_LIGHTING_LABELS = [
    "dark_place",
    "bright_place",
    "day",
    "night",
]

SCENE_COMPOSITION_LABELS = [
    "close_up",
    "medium_shot",
    "wide_shot",
    "empty_scene",
    "one_person",
    "two_people",
    "group_of_people",
    "crowded_scene",
]

SCENE_STATE_LABELS = [
    "calm_scene",
    "action_scene",
    "conversation_scene",
    "movement_scene",
]

SCENE_ACTION_HINT_LABELS = [
    "walking",
    "running",
    "sitting",
    "standing",
    "working",
    "playing",
    "driving",
    "dancing",
    "eating",
    "drinking",
]

SPECTRA_SCENE_LABELS = (
    SCENE_ENVIRONMENT_LABELS
    + SCENE_LIGHTING_LABELS
    + SCENE_COMPOSITION_LABELS
    + SCENE_STATE_LABELS
    + SCENE_ACTION_HINT_LABELS
)


# ============================================================
# 2. Labels do modelo de pessoa
# ============================================================

PERSON_BODY_LABELS = [
    "person",
    "face_visible",
    "hand_visible",
    "man",
    "woman",
    "child",
]

PERSON_HAIR_LABELS = [
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

PERSON_CLOTHING_LABELS = [
    "red_clothes",
    "blue_clothes",
    "black_clothes",
    "white_clothes",
    "green_clothes",
    "yellow_clothes",
    "dress",
    "shirt",
    "jacket",
]

PERSON_ACCESSORY_LABELS = [
    "glasses",
    "hat",
    "cap",
    "backpack",
    "bag",
]

PERSON_SKIN_TONE_LABELS = [
    "light_skin_tone",
    "medium_skin_tone",
    "dark_skin_tone",
]

SPECTRA_PERSON_LABELS = (
    PERSON_BODY_LABELS
    + PERSON_HAIR_LABELS
    + PERSON_CLOTHING_LABELS
    + PERSON_ACCESSORY_LABELS
    + PERSON_SKIN_TONE_LABELS
)


# ============================================================
# 3. Labels do modelo de objetos
# ============================================================

OBJECT_COMMON_LABELS = [
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
    "box",
    "paper",
    "toy",
]

OBJECT_NARRATIVE_LABELS = [
    "glasses",
    "knife",
    "weapon",
    "key",
    "letter",
    "document",
    "photo",
    "blood",
    "bag",
    "backpack",
    "book",
    "phone",
]

OBJECT_SPECIFIC_LABELS = [
    "dice",
    "miniature",
    "board_game",
    "musical_instrument",
    "subtitles",
    "on_screen_text",
]

SPECTRA_OBJECT_LABELS = (
    OBJECT_COMMON_LABELS
    + OBJECT_NARRATIVE_LABELS
    + OBJECT_SPECIFIC_LABELS
)

# Remove duplicadas preservando ordem.
SPECTRA_OBJECT_LABELS = list(dict.fromkeys(SPECTRA_OBJECT_LABELS))


# ============================================================
# 4. Labels gerais antigas / compatibilidade
# ============================================================

SPECTRA_LABELS = list(dict.fromkeys(
    SPECTRA_SCENE_LABELS
    + SPECTRA_PERSON_LABELS
    + SPECTRA_OBJECT_LABELS
))


SPECTRA_LABEL_GROUPS = {
    "scene": SPECTRA_SCENE_LABELS,
    "person": SPECTRA_PERSON_LABELS,
    "object": SPECTRA_OBJECT_LABELS,
    "all": SPECTRA_LABELS,
}


def get_labels_for_task(task_name):
    """
    Retorna as labels de acordo com o submodelo.

    task_name:
        - scene
        - person
        - object
        - all
    """
    if task_name not in SPECTRA_LABEL_GROUPS:
        raise ValueError(
            "Task inválida: {}. Opções: {}".format(
                task_name,
                list(SPECTRA_LABEL_GROUPS.keys())
            )
        )

    return SPECTRA_LABEL_GROUPS[task_name]


def get_label_count(task_name="all"):
    return len(get_labels_for_task(task_name))


def split_predictions_by_group(predictions):
    grouped_predictions = {
        "scene": [],
        "person": [],
        "object": [],
    }

    for prediction in predictions:
        label = prediction["label"]

        if label in SPECTRA_SCENE_LABELS:
            grouped_predictions["scene"].append(prediction)

        if label in SPECTRA_PERSON_LABELS:
            grouped_predictions["person"].append(prediction)

        if label in SPECTRA_OBJECT_LABELS:
            grouped_predictions["object"].append(prediction)

    return grouped_predictions