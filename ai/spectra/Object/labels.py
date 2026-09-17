OBJECT_FURNITURE_LABELS = [
    "chair",
    "table",
    "sofa",
    "bed",
    "toilet",
    "door",
    "window",
    "mirror",
    "desk",
    "shelf",
    "cabinet",
    "lamp",
]

OBJECT_ELECTRONIC_LABELS = [
    "phone",
    "computer",
    "screen",
    "television",
    "keyboard",
    "mouse",
    "remote",
    "camera",
    "headphones",
    "microphone",
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
    "fish",
]

OBJECT_FOOD_KITCHEN_LABELS = [
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

OBJECT_BAG_ACCESSORY_LABELS = [
    "bag",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "glasses",
    "watch",
    "hat",
]

OBJECT_READING_LABELS = [
    "book",
    "paper",
    "letter",
    "document",
    "photo",
    "newspaper",
    "poster",
]

OBJECT_SPORT_TOY_LABELS = [
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

LABELS = list(dict.fromkeys(
    OBJECT_FURNITURE_LABELS
    + OBJECT_ELECTRONIC_LABELS
    + OBJECT_VEHICLE_LABELS
    + OBJECT_ANIMAL_LABELS
    + OBJECT_FOOD_KITCHEN_LABELS
    + OBJECT_BAG_ACCESSORY_LABELS
    + OBJECT_READING_LABELS
    + OBJECT_SPORT_TOY_LABELS
    + OBJECT_NARRATIVE_LABELS
))

SPECTRA_OBJECT_LABELS = LABELS