from pipeline.orchestration.process_video import process_video

result = process_video(
    video_path="data/raw_videos/video_teste.mp4",
    output_base_dir="data/output",

    scene_model_path="data/models/Scene/scene_net_best.pt",
    person_model_path="data/models/Person_v2_age/person_net_best.pt",
    object_model_path=None,
    action_model_path="data/models/Actions_v2_full_finetuned/action_net_best.pt",

    run_spectra=True,
    run_narrative=False,
    run_tts=False,

    spectra_scene_threshold=0.45,
    spectra_action_threshold=0.3,
    spectra_top_k=10,

    use_person_cropper=True,
    use_action_model=True,
    use_action_person_cropper=True,
    action_max_people=5,
)

print("Pipeline finalizado.")
print(result["artifacts"])