from pathlib import Path

import pandas as pd


def main():
    root = Path("data/external/celeba/celeba")

    if not root.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {root}")

    attr_path = root / "list_attr_celeba.txt"
    images_dir = root / "img_align_celeba"

    print("Raiz CelebA:")
    print(root.resolve())

    print("\nArquivo de atributos existe?", attr_path.exists())
    print("Pasta de imagens existe?", images_dir.exists())

    image_paths = sorted(images_dir.glob("*.jpg")) if images_dir.exists() else []

    print("\nTotal de imagens:", len(image_paths))

    if image_paths:
        print("Primeiras imagens:")
        for path in image_paths[:10]:
            print("-", path)

    if attr_path.exists():
        df = pd.read_csv(
            attr_path,
            sep=r"\s+",
            skiprows=1,
            engine="python",
        )

        print("\nShape atributos:", df.shape)

        print("\nColunas:")
        print(list(df.columns))

        print("\nPrimeiras linhas:")
        print(df.head())


if __name__ == "__main__":
    main()