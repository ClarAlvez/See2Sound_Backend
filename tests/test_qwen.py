
from ai.narrative.narrative_generator import LLMNarrativeGenerator
from ai.narrative.data_models import NarrativeInput

generator = LLMNarrativeGenerator(use_llm=False)

samples = [
    NarrativeInput(
        labels=[
            "person", "woman", "long_hair",
            "backpack", "umbrella", "outdoor"
        ],
        confidence={
            "person.woman": 0.99,
            "person.long_hair": 0.96,
            "object.backpack": 0.95,
            "object.umbrella": 0.90,
            "scene.outdoor": 0.98,
        },
    ),
    NarrativeInput(
        labels=[
            "person", "walking", "street",
            "black_clothes"
        ],
        confidence={
            "person.person": 1.0,
            "action.walking": 0.92,
            "scene.street": 0.91,
            "person.black_clothes": 0.85,
        },
    ),
    NarrativeInput(
        labels=[
            "umbrella", "window", "running",
            "outdoor"
        ],
        confidence={
            "object.umbrella": 0.94,
            "object.window": 0.90,
            "action.running": 0.95,
            "scene.outdoor": 0.98,
        },
    ),
]

for item in samples:
    output = generator.generate(item)

    print("\nLabels:", item.labels)
    print("Fatos:", output.scene_context)
    print("Descrição:", output.description)
    print("Ignorada:", output.skipped)
