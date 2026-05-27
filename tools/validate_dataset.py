import pandas as pd

csv_path = "data/datasets/spectra_object_labels.csv"

df = pd.read_csv(csv_path)

metadata_cols = [
    "frame_path",
    "source",
    "source_dataset",
    "source_query",
    "source_video_id",
    "source_page_url",
    "source_sample_path",
    "source_label",
    "source_sample_id",
]

label_cols = [c for c in df.columns if c not in metadata_cols]

counts = df[label_cols].sum().sort_values(ascending=False)

print("Total de imagens/crops:", len(df))
print("\nLabels mais frequentes:")
print(counts.head(40))

print("\nLabels zeradas:")
print(list(counts[counts == 0].index))

print("\nMédia de labels positivas por imagem:")
print(df[label_cols].sum(axis=1).mean())