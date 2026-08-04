from ai.spectra.predictor import SpectraPredictor

predictor = SpectraPredictor(
    model_path="data/models/spectra_scene/scene_net_best.pt",
    threshold=0.1,
    top_k=10,
    task_name="scene",
)

result = predictor.predict_frame(
    image_path="data/output/frames/video_teste_frame_000300.jpg",
    group_by_category=True,
)

for prediction in result["predictions"]:
    print(f"{prediction['label']}: {prediction['score']:.2f}")
