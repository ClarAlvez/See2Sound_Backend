from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main():
    mat_path = Path("data/external/market1501/market_attribute.mat")

    if not mat_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {mat_path}\n"
            "Coloque o market_attribute.mat em data/external/market1501/"
        )

    data = loadmat(str(mat_path))

    print("Arquivo encontrado:", mat_path)
    print("\nChaves principais:")

    for key in data.keys():
        if not key.startswith("__"):
            print("-", key)

    if "market_attribute" not in data:
        raise ValueError(
            "O arquivo .mat não possui a chave 'market_attribute'. "
            "Talvez esse não seja o market_attribute.mat correto."
        )

    market = data["market_attribute"][0, 0]

    print("\nEstrutura de market_attribute:")
    print("type:", type(market))
    print("dtype:", getattr(market, "dtype", None))

    if not hasattr(market, "dtype") or not market.dtype.names:
        raise ValueError("market_attribute não possui campos nomeados.")

    print("fields:", market.dtype.names)

    for split_name in ["train", "test"]:
        if split_name not in market.dtype.names:
            print(f"\nSplit '{split_name}' não encontrado.")
            continue

        split = market[split_name][0, 0]

        print("\n" + "=" * 80)
        print("SPLIT:", split_name)
        print("type:", type(split))
        print("shape:", getattr(split, "shape", None))
        print("dtype:", getattr(split, "dtype", None))

        if hasattr(split, "dtype") and split.dtype.names:
            print("\nCampos:")

            for field in split.dtype.names:
                value = split[field]

                print(
                    "-",
                    field,
                    "| shape:",
                    getattr(value, "shape", None),
                    "| dtype:",
                    getattr(value, "dtype", None),
                )

                try:
                    unwrapped = value

                    while hasattr(unwrapped, "shape") and unwrapped.shape == (1, 1):
                        unwrapped = unwrapped[0, 0]

                    example = np.array(unwrapped).flatten()[:10]
                    print("  exemplo:", example)

                except Exception as error:
                    print("  erro ao ler exemplo:", error)


if __name__ == "__main__":
    main()