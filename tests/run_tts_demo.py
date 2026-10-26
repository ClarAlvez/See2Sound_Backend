from pathlib import Path
from tempfile import mkdtemp


def main():
    from ai.audio_description.tts_client import TTSClient
    tts = TTSClient(
        rate=170,
        volume=1.0,
    )

    print("Vozes disponíveis:")
    for voice in tts.list_voices():
        print(voice)

    output = tts.save_to_file(
        text="Uma pessoa corre pela rua durante a noite.",
        output_path=str(Path(mkdtemp(prefix="see2sound_tts_demo_")) / "test_tts.wav"),
    )

    print("Áudio gerado em:", output)


if __name__ == "__main__":
    main()
