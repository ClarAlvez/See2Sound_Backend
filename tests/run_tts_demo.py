from ai.audio_description.tts_client import TTSClient


def main():
    tts = TTSClient(
        rate=170,
        volume=1.0,
    )

    print("Vozes disponíveis:")
    for voice in tts.list_voices():
        print(voice)

    output = tts.save_to_file(
        text="Uma pessoa corre pela rua durante a noite.",
        output_path="data/outputs/audio_descriptions/test_tts.wav",
    )

    print("Áudio gerado em:", output)


if __name__ == "__main__":
    main()
