"""Manual example; import and pytest collection do not execute it."""


def main():
    from pipeline.video.metadata import get_video_metadata

    video_path = "data/raw_videos/novo.mp4"

    print("\nTeste de extração de metadata:")

    metadata = get_video_metadata(video_path)
    print(metadata)


if __name__ == "__main__":
    main()
