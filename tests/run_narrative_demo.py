from ai.narrative import LLMNarrativeGenerator


def main():
    generator = LLMNarrativeGenerator(
        model_path="data/models/llama/Llama-3.2-1B-Instruct-Q6_K_L.gguf",
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
    )

    spectra_outputs = [
        {
            "start_time": 0.0,
            "end_time": 4.0,
            "labels": ["person", "running", "street", "night"],
        },
        {
            "start_time": 4.0,
            "end_time": 8.0,
            "labels": ["person", "running", "street", "night"],
        },
        {
            "start_time": 8.0,
            "end_time": 12.0,
            "labels": ["car", "driving", "road", "night"],
        },
    ]

    timeline = generator.generate_timeline_from_dicts(spectra_outputs)

    for item in timeline:
        print("-" * 50)
        print("Tempo: {} -> {}".format(item["start_time"], item["end_time"]))
        print("Descrição:", item["description"])
        print("Labels:", item["labels"])

        if item["fidelity_warnings"]:
            print("Avisos:", item["fidelity_warnings"])


if __name__ == "__main__":
    main()
