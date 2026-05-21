import pandas as pd

csv_path = "data/datasets/spectra_auto_labels_margin_v2.csv"

df = pd.read_csv(csv_path)

metadata_cols = [
    "frame_path",
    "source",
    "source_query",
    "source_video_id",
    "source_page_url",
]

label_cols = [c for c in df.columns if c not in metadata_cols]

counts = df[label_cols].sum().sort_values(ascending=False)
total = len(df)

print("Total de frames:", total)

print("\nLabels mais frequentes:")
print(counts.head(40))

print("\nLabels zeradas:")
print(list(counts[counts == 0].index))

print("\nLabels presentes em todos os frames:")
print(list(counts[counts == total].index))

print("\nMédia de labels positivas por frame:")
print(df[label_cols].sum(axis=1).mean())

print("\nDistribuição de labels positivas por frame:")
print(df[label_cols].sum(axis=1).describe())