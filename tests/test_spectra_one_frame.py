from ai.spectra.predictor import SpectraPredictor

predictor = SpectraPredictor(
    model_path="data/models/spectra_net.pt",
    threshold=0.1
)

result = predictor.predict_frame(
    "data/test_frames/video_teste_cachorro.jpg",
    top_k=10
)

for prediction in result["predictions"]:
    print(f"{prediction['label']}: {prediction['score']:.2f}")