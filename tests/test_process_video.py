from pipeline.orchestration.process_video import process_video

result = process_video(
    video_path="data/raw_videos/video_teste.mp4",
    output_base_dir="data/output",

    scene_model_path="data/models/spectra_scene/scene_net_best.pt",
    person_model_path="data/models/spectra_person_v3_hair/person_net_best.pt",
    object_model_path=None,

    spectra_scene_threshold=0.45,
    spectra_person_threshold=0.45,
    spectra_top_k=20,

    use_person_cropper=True,
    use_person_model_on_full_frame=False,

    run_spectra=True,
    run_narrative=False,
    run_tts=False,
)

print("Pipeline finalizado.")
print(result["artifacts"])