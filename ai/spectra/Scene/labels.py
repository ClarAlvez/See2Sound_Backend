SCENE_ENVIRONMENT_LABELS = [
    "indoor", "outdoor", "room", "street", "school", "classroom", "kitchen",
    "bedroom", "living_room", "office", "forest", "city", "sports_field",
    "beach", "park", "store", "restaurant", "hospital",
]
SCENE_LIGHTING_LABELS = ["dark_place", "bright_place", "day", "night"]
SCENE_COMPOSITION_LABELS = [
    "close_up", "medium_shot", "wide_shot", "empty_scene", "one_person",
    "two_people", "group_of_people", "crowded_scene",
]
SCENE_STATE_LABELS = [
    "calm_scene", "action_scene", "conversation_scene", "movement_scene",
]
SCENE_ACTION_HINT_LABELS = [
    "walking", "running", "sitting", "standing", "working", "playing",
    "driving", "dancing", "eating", "drinking",
]
SPECTRA_SCENE_LABELS = (
    SCENE_ENVIRONMENT_LABELS + SCENE_LIGHTING_LABELS + SCENE_COMPOSITION_LABELS
    + SCENE_STATE_LABELS + SCENE_ACTION_HINT_LABELS
)
LABELS = SPECTRA_SCENE_LABELS
