from pathlib import Path

import pandas as pd


def main():
    root = Path("data/external/upar")

    if not root.exists():
        raise FileNotFoundError("Pasta não encontrada: data/external/upar")

    print("\nRaiz inspecionada:")
    print(root.resolve())

    print("\n" + "=" * 100)
    print("Arquivos CSV encontrados:")
    csv_paths = sorted(root.rglob("*.csv"))

    if not csv_paths:
        print("Nenhum CSV encontrado.")
    else:
        for path in csv_paths:
            print("-", path)

    print("\n" + "=" * 100)
    print("Arquivos JSON encontrados:")
    json_paths = sorted(root.rglob("*.json"))

    if not json_paths:
        print("Nenhum JSON encontrado.")
    else:
        for path in json_paths:
            print("-", path)

    print("\n" + "=" * 100)
    print("Arquivos PKL encontrados:")
    pkl_paths = sorted(root.rglob("*.pkl"))

    if not pkl_paths:
        print("Nenhum PKL encontrado.")
    else:
        for path in pkl_paths:
            print("-", path)

    print("\n" + "=" * 100)
    print("Arquivos MAT encontrados:")
    mat_paths = sorted(root.rglob("*.mat"))

    if not mat_paths:
        print("Nenhum MAT encontrado.")
    else:
        for path in mat_paths:
            print("-", path)

    print("\n" + "=" * 100)
    print("Arquivos TXT encontrados:")
    txt_paths = sorted(root.rglob("*.txt"))

    if not txt_paths:
        print("Nenhum TXT encontrado.")
    else:
        for path in txt_paths[:50]:
            print("-", path)

        if len(txt_paths) > 50:
            print(f"... mais {len(txt_paths) - 50} arquivos TXT")

    print("\n" + "=" * 100)
    print("Arquivos de imagem encontrados:")

    image_paths = (
        list(root.rglob("*.jpg"))
        + list(root.rglob("*.jpeg"))
        + list(root.rglob("*.png"))
        + list(root.rglob("*.bmp"))
    )

    print("Total de imagens:", len(image_paths))

    if image_paths:
        print("Exemplos:")
        for path in image_paths[:20]:
            print("-", path)

    print("\n" + "=" * 100)
    print("Prévia dos CSVs:")

    for csv_path in csv_paths[:20]:
        print("\n" + "=" * 100)
        print("CSV:", csv_path)

        try:
            df = pd.read_csv(csv_path)
        except Exception as error:
            print("Erro lendo CSV:", error)
            continue

        print("Shape:", df.shape)

        print("\nColunas:")
        print(list(df.columns))

        print("\nPrimeiras linhas:")
        print(df.head())

    print("\n" + "=" * 100)
    print("Inspeção concluída.")


if __name__ == "__main__":
    main()