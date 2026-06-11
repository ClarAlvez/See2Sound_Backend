from pipeline.orchestration.process_video import process_video

result = process_video(
    video_path="data/raw_videos/video_teste.mp4",
    output_base_dir="data/output",

    scene_model_path="data/models/spectra_scene/scene_net_best.pt",
    person_model_path=None,
    object_model_path=None,

    tts_rate=170,
    tts_volume=1.0,

    run_spectra=True,
    run_narrative=True,
    run_tts=True,
)