"""
Labels do modelo Object da Spectra.

A ideia é manter labels úteis para audiodescrição e, ao mesmo tempo,
compatíveis com um primeiro treino em COCO/Open Images.
"""

OBJECT_FURNITURE_LABELS = [
    "chair",
    "table",
    "sofa",
    "bed",
    "toilet",
]

OBJECT_STRUCTURE_LABELS = [
    "door",
    "window",
]

OBJECT_ELECTRONICS_LABELS = [
    "phone",
    "computer",
    "screen",
    "television",
    "keyboard",
    "mouse",
    "remote",
]

OBJECT_VEHICLE_LABELS = [
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "boat",
    "airplane",
]

OBJECT_ANIMAL_LABELS = [
    "animal",
    "dog",
    "cat",
    "bird",
    "horse",
    "sheep",
    "cow",
]

OBJECT_FOOD_AND_KITCHEN_LABELS = [
    "food",
    "fruit",
    "cup",
    "bottle",
    "bowl",
    "plate",
    "fork",
    "spoon",
    "knife",
]

OBJECT_PERSONAL_LABELS = [
    "bag",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "glasses",
]

OBJECT_DOCUMENT_LABELS = [
    "book",
    "paper",
    "letter",
    "document",
    "photo",
]

OBJECT_SPORT_AND_PLAY_LABELS = [
    "ball",
    "toy",
    "kite",
    "skateboard",
    "surfboard",
    "sports_racket",
]

OBJECT_NARRATIVE_LABELS = [
    "box",
    "key",
    "weapon",
    "blood",
    "musical_instrument",
    "dice",
    "miniature",
    "board_game",
    "subtitles",
    "on_screen_text",
]

SPECTRA_OBJECT_LABELS = list(
    dict.fromkeys(
        OBJECT_FURNITURE_LABELS
        + OBJECT_STRUCTURE_LABELS
        + OBJECT_ELECTRONICS_LABELS
        + OBJECT_VEHICLE_LABELS
        + OBJECT_ANIMAL_LABELS
        + OBJECT_FOOD_AND_KITCHEN_LABELS
        + OBJECT_PERSONAL_LABELS
        + OBJECT_DOCUMENT_LABELS
        + OBJECT_SPORT_AND_PLAY_LABELS
        + OBJECT_NARRATIVE_LABELS
    )
)

LABELS = SPECTRA_OBJECT_LABELS
