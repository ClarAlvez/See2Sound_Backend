from pathlib import Path
import shutil

import pandas as pd


INPUT_CSV = Path("data/datasets/Actions/action_training_v3_full.csv")
OUTPUT_DIR = Path("data/kaggle/spectra_actions_v3_training")
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_CSV = OUTPUT_DIR / "spectra_scene_subcategories_debug.csv"


def main():
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    new_paths = []

    copied = 0
    missing = 0

    for index, row in df.iterrows():
        original_path = Path(str(row["frame_path"]))

        if not original_path.exists():
            print("Imagem não encontrada:", original_path)
            new_paths.append("")
            missing += 1
            continue

        new_name = f"{index:06d}_{original_path.name}"
        destination_path = OUTPUT_IMAGES_DIR / new_name

        if not destination_path.exists():
            shutil.copy2(original_path, destination_path)
            copied += 1

        new_paths.append(f"images/{new_name}")

    df["frame_path"] = new_paths
    df = df[df["frame_path"] != ""].copy()

    df.to_csv(OUTPUT_CSV, index=False)

    print("Exportação finalizada.")
    print("Pasta criada:", OUTPUT_DIR)
    print("CSV exportado:", OUTPUT_CSV)
    print("Imagens copiadas:", copied)
    print("Imagens faltando:", missing)
    print("Total final:", len(df))


if __name__ == "__main__":
    main()  