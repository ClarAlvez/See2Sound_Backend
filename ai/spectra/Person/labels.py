PERSON_BODY_LABELS = ["person", "man", "woman"]
PERSON_HAIR_LABELS = [
    "short_hair", "long_hair", "bald_hair", "black_hair", "blonde_hair",
    "brown_hair", "gray_hair", "straight_hair", "wavy_hair", "bangs_hair",
    "receding_hairline",
]
PERSON_CLOTHING_COLOR_LABELS = [
    "black_clothes", "white_clothes", "red_clothes", "blue_clothes",
    "green_clothes", "yellow_clothes", "brown_clothes", "gray_clothes",
    "orange_clothes", "pink_clothes", "purple_clothes",
]
PERSON_CLOTHING_TYPE_LABELS = ["dress"]
PERSON_ACCESSORY_LABELS = ["glasses", "hat", "backpack", "bag"]
SPECTRA_PERSON_LABELS = (
    PERSON_BODY_LABELS + PERSON_HAIR_LABELS + PERSON_CLOTHING_COLOR_LABELS
    + PERSON_CLOTHING_TYPE_LABELS + PERSON_ACCESSORY_LABELS
)
LABELS = SPECTRA_PERSON_LABELS
