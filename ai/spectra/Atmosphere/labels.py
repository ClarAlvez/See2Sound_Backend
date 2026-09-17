LABELS = [
    # Horário / período
    "day",
    "night",
    "dawn_dusk",

    # Iluminação
    "bright",
    "low_light",

    # Condições atmosféricas
    "clear_weather",
    "sunny",
    "cloudy",
    "foggy",
    "rainy",
    "snowy",
]


LABEL_GROUPS = {
    "time_of_day": [
        "day",
        "night",
        "dawn_dusk",
    ],

    "lighting": [
        "bright",
        "low_light",
    ],

    "weather": [
        "clear_weather",
        "sunny",
        "cloudy",
        "foggy",
        "rainy",
        "snowy",
    ],
}


EXCLUSIVE_GROUPS = [
    [
        "day",
        "night",
        "dawn_dusk",
    ],
]

ATMOSPHERE_LABELS = LABELS