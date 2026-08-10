OBJECT_COMMON_LABELS = [
    "book", "table", "chair", "sofa", "bed", "door", "window", "phone",
    "computer", "screen", "television", "car", "bicycle", "motorcycle",
    "bus", "animal", "dog", "cat", "food", "cup", "bottle", "box",
    "paper", "toy",
]
OBJECT_NARRATIVE_LABELS = [
    "glasses", "knife", "weapon", "key", "letter", "document", "photo",
    "blood", "bag", "backpack", "book", "phone",
]
OBJECT_SPECIFIC_LABELS = [
    "dice", "miniature", "board_game", "musical_instrument", "subtitles",
    "on_screen_text",
]
SPECTRA_OBJECT_LABELS = list(dict.fromkeys(
    OBJECT_COMMON_LABELS + OBJECT_NARRATIVE_LABELS + OBJECT_SPECIFIC_LABELS
))
LABELS = SPECTRA_OBJECT_LABELS
