POSE_LABELS = [
    "standing",
    "sitting",
    "lying_down",
    "crouching",
    "arms_raised",
]

LOCOMOTION_LABELS = [
    "walking",
    "running",
    "jumping",
    "cycling",
    "driving",
    "swimming",
    "climbing",
]

ACTIVITY_LABELS = [
    "dancing",
    "playing",
    "working",
    "eating",
    "drinking",
    "exercising",
    "sports",
    "instrument_playing",
    "phone_use",
    "computer_use",
    "reading",
    "writing",
    "cooking",
    "cleaning",
    "makeup",
    "grooming",
    "talking",
    "reaching",
    "throwing",
    "carrying",
    "radio_use",
    "ball_sport",
    "racket_sport",
    "martial_activity",
    "water_activity",
]

MOVEMENT_STATE_LABELS = [
    "still",
    "moving",
    "fast_motion",
    "falling",
]

LABELS = (
    POSE_LABELS
    + LOCOMOTION_LABELS
    + ACTIVITY_LABELS
    + MOVEMENT_STATE_LABELS
)