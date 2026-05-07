OBJECT_LABELS = [
    "person",
    "book",
    "table",
    "chair",
    "sofa",
    "door",
    "window",
    "phone",
    "computer",
    "screen",
    "car",
    "animal",
    "food",
    "cup",
    "bag",
    "weapon",
    "knife",
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
    "pointing",
    "smiling",
    "crying",
    "opening",
    "showing",
]

SCENARIO_LABELS = [
    "indoor",
    "outdoor",
    "room",
    "street",
    "school",
    "kitchen",
    "bedroom",
    "living_room",
    "forest",
    "city",
    "sports_field",
    "dark_place",
    "bright_place",
    "day",
    "night",
]

COMPOSITION_LABELS = [
    "close_up",
    "medium_shot",
    "wide_shot",
    "one_person",
    "two_people",
    "group_of_people",
    "crowded_scene",
    "empty_scene",
]

EXPRESSION_LABELS = [
    "happy_expression",
    "serious_expression",
    "surprised_expression",
    "sad_expression",
]

SPECTRA_LABELS = (
    OBJECT_LABELS
    + ACTION_LABELS
    + SCENARIO_LABELS
    + COMPOSITION_LABELS
    + EXPRESSION_LABELS
)