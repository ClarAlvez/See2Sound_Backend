SPECTRA_SCENE_LABELS = [
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

    # Office subcategorias
    "office_room",
    "office_cubicles",
    "home_office",
    "conference_room",

    # Restaurant subcategorias
    "restaurant_indoor",
    "fastfood_restaurant",
    "cafeteria",
    "dining_room",
    "restaurant_patio",

    "store",
    "hospital",
    "sports_field",
]

LABELS = SPECTRA_SCENE_LABELS

DERIVED_SCENE_GROUPS = {
    "office": [
        "office_room",
        "office_cubicles",
        "home_office",
        "conference_room",
    ],
    "restaurant": [
        "restaurant_indoor",
        "fastfood_restaurant",
        "cafeteria",
        "dining_room",
        "restaurant_patio",
    ],
}