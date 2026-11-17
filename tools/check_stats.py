import pandas as pd

df = pd.read_csv(
    "data/datasets/spectra_atmosphere_v4_masked.csv"
)

print(pd.crosstab(
    df["source_dataset"],
    df["generated_split"]
))

print("\nBDD - origem das amostras:")
bdd = df[df["source_dataset"] == "bdd100k"]

print(pd.crosstab(
    bdd["source_split"],
    bdd["generated_split"]
))