from pipeline.orchestration.process_video import process_video


VIDEO_PATH = "data/raw_videos/video_teste.mp4"

SCENE_MODEL_PATH = "data/models/Scene/scene_net_best.pt"

PERSON_MODEL_PATH = "data/models/Person/person_net_best.pt"

ACTION_MODEL_PATH = "data/models/Actions/action_net_best.pt"


NARRATIVE_MODEL_PATH = (
    "data/models/llama/Llama-3.2-1B-Instruct-Q6_K_L.gguf"
)


def main():
    print("=" * 80)
    print("TESTE SEE2SOUND")
    print("Spectra -> Narrative")
    print("=" * 80)

    result = process_video(
        video_path=VIDEO_PATH,
        output_base_dir="data/output",

        scene_model_path=SCENE_MODEL_PATH,
        person_model_path=PERSON_MODEL_PATH,
        object_model_path=None,
        action_model_path=ACTION_MODEL_PATH,

        # Modelo Llama local
        narrative_model_path=NARRATIVE_MODEL_PATH,

        run_spectra=True,
        run_narrative=True,
        run_tts=False,

        spectra_scene_threshold=0.45,
        spectra_action_threshold=0.3,
        spectra_top_k=10,

        use_person_cropper=True,
        use_action_model=True,
        use_action_person_cropper=True,
        action_max_people=5,
    )

    print("\n")
    print("=" * 80)
    print("SPECTRA")
    print("=" * 80)

    spectra_outputs = result.get("spectra", {}).get("outputs", [])

    if not spectra_outputs:
        print("Nenhuma saída da Spectra.")
    else:
        for index, scene in enumerate(spectra_outputs, start=1):
            print(f"\nCena {index}")
            print(
                f"Tempo: "
                f"{scene.get('start_time', 0):.2f}s -> "
                f"{scene.get('end_time', 0):.2f}s"
            )

            print("Labels:")

            confidence = scene.get("confidence", {})

            for label in scene.get("labels", []):
                score = None

                for key, value in confidence.items():
                    if key.endswith(f".{label}") or key == label:
                        score = value
                        break

                if score is not None:
                    print(f"  - {label}: {score:.4f}")
                else:
                    print(f"  - {label}")

    print("\n")
    print("=" * 80)
    print("NARRATIVE")
    print("=" * 80)

    timeline = result.get("narrative", {}).get("timeline", [])

    if not timeline:
        print("Nenhuma descrição narrativa foi gerada.")
        return

    for index, item in enumerate(timeline, start=1):
        print(f"\nDescrição {index}")

        start_time = item.get("start_time", 0.0)
        end_time = item.get("end_time", start_time)

        print(
            f"Tempo: "
            f"{start_time:.2f}s -> "
            f"{end_time:.2f}s"
        )

        description = item.get("description", "")

        if description:
            print("Texto:")
            print(f"  {description}")
        else:
            print("Texto: [vazio]")

        labels = item.get("labels", [])

        if labels:
            print("Labels utilizadas:")
            print("  " + ", ".join(labels))

        if item.get("skipped"):
            print(
                "Ignorada:",
                item.get("skip_reason", "sem motivo informado"),
            )

        warnings = item.get("fidelity_warnings")

        if warnings:
            print("Avisos de fidelidade:")

            for warning in warnings:
                print(f"  - {warning}")

    print("\n")
    print("=" * 80)
    print("DESCRIÇÕES FINAIS")
    print("=" * 80)

    valid_descriptions = [
        item["description"]
        for item in timeline
        if item.get("description")
        and not item.get("skipped", False)
    ]

    for description in valid_descriptions:
        print(f"- {description}")

    print("\nTeste concluído.")


if __name__ == "__main__":
    main()