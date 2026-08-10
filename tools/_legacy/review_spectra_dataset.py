import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from ai.spectra.Object.labels import SPECTRA_OBJECT_LABELS
from ai.spectra.Person.labels import LABELS
from ai.spectra.Scene.labels import SPECTRA_SCENE_LABELS

SPECTRA_LABEL_GROUPS = {
    "scene": SPECTRA_SCENE_LABELS,
    "person": LABELS,
    "object": SPECTRA_OBJECT_LABELS,
}
SPECTRA_LABELS = list(dict.fromkeys(
    SPECTRA_SCENE_LABELS + LABELS + SPECTRA_OBJECT_LABELS
))


METADATA_COLUMNS = [
    "frame_path",
    "source",
    "source_query",
    "source_video_id",
    "source_page_url",
]


HELP_TEXT = """
Comandos disponíveis:

Enter
    Mantém o frame como está e vai para o próximo.

a label1,label2,label3
    Adiciona labels.
    Exemplo: a person,walking,outdoor

r label1,label2,label3
    Remove labels.
    Exemplo: r backpack,knife

set label=1,label2=0
    Define valores diretamente.
    Exemplo: set person=1,empty_scene=0

clear
    Zera todas as labels do frame atual.

drop
    Marca o frame como não utilizável no treino.

keep
    Marca o frame como utilizável no treino.

labels
    Mostra todas as labels disponíveis.

groups
    Mostra labels por grupo.

active
    Mostra as labels ativas no frame atual.

scores
    Mostra os maiores scores do CSV de scores, se informado.

back
    Volta um frame.

q
    Salva e sai.
"""


def load_or_create_review_csv(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    if output_csv.exists():
        print(f"Carregando revisão existente: {output_csv}")
        return pd.read_csv(output_csv)

    print(f"Criando CSV revisado a partir de: {input_csv}")
    shutil.copy(input_csv, output_csv)

    dataframe = pd.read_csv(output_csv)

    if "reviewed" not in dataframe.columns:
        dataframe["reviewed"] = 0

    if "use_frame" not in dataframe.columns:
        dataframe["use_frame"] = 1

    dataframe.to_csv(output_csv, index=False)

    return dataframe


def validate_label_columns(dataframe: pd.DataFrame) -> None:
    missing_labels = [
        label for label in SPECTRA_LABELS
        if label not in dataframe.columns
    ]

    if missing_labels:
        raise ValueError(
            "O CSV não possui as seguintes labels esperadas: {}".format(
                missing_labels
            )
        )


def open_image(image_path: Path) -> None:
    if not image_path.exists():
        print(f"Imagem não encontrada: {image_path}")
        return

    try:
        image = Image.open(image_path)
        image.show()
    except Exception as error:
        print(f"Não foi possível abrir a imagem: {error}")


def get_active_labels(row) -> list:
    active_labels = []

    for label in SPECTRA_LABELS:
        if int(row[label]) == 1:
            active_labels.append(label)

    return active_labels


def parse_label_list(raw_text: str) -> list:
    labels = []

    for part in raw_text.split(","):
        label = part.strip()

        if label:
            labels.append(label)

    return labels


def validate_labels(labels: list) -> list:
    invalid_labels = [
        label for label in labels
        if label not in SPECTRA_LABELS
    ]

    return invalid_labels


def print_frame_info(dataframe: pd.DataFrame, index: int) -> None:
    row = dataframe.iloc[index]

    frame_path = row["frame_path"]
    use_frame = int(row.get("use_frame", 1))
    reviewed = int(row.get("reviewed", 0))

    active_labels = get_active_labels(row)

    print("\n" + "=" * 80)
    print(f"Frame {index + 1}/{len(dataframe)}")
    print(f"Caminho: {frame_path}")
    print(f"use_frame: {use_frame} | reviewed: {reviewed}")

    if "source_query" in dataframe.columns:
        print(f"source_query: {row.get('source_query')}")

    if "source_video_id" in dataframe.columns:
        print(f"source_video_id: {row.get('source_video_id')}")

    print("\nLabels ativas:")
    if active_labels:
        print(", ".join(active_labels))
    else:
        print("(nenhuma)")

    print("=" * 80)


def print_all_labels() -> None:
    print("\nTodas as labels disponíveis:")

    for label in SPECTRA_LABELS:
        print(f"- {label}")


def print_groups() -> None:
    print("\nLabels por grupo:")

    for group_name, labels in SPECTRA_LABEL_GROUPS.items():
        print(f"\n[{group_name}]")
        print(", ".join(labels))


def print_active_labels(dataframe: pd.DataFrame, index: int) -> None:
    row = dataframe.iloc[index]
    active_labels = get_active_labels(row)

    print("\nLabels ativas:")

    if active_labels:
        print(", ".join(active_labels))
    else:
        print("(nenhuma)")


def print_scores(scores_dataframe, index: int, top_k: int = 20) -> None:
    if scores_dataframe is None:
        print("\nNenhum CSV de scores foi informado.")
        return

    row = scores_dataframe.iloc[index]

    scores = []

    for label in SPECTRA_LABELS:
        if label in scores_dataframe.columns:
            try:
                scores.append((label, float(row[label])))
            except (TypeError, ValueError):
                pass

    scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    print(f"\nTop {top_k} scores:")

    for label, score in scores[:top_k]:
        print(f"- {label}: {score:.4f}")


def add_labels(dataframe: pd.DataFrame, index: int, labels: list) -> None:
    invalid_labels = validate_labels(labels)

    if invalid_labels:
        print(f"Labels inválidas: {invalid_labels}")
        return

    for label in labels:
        dataframe.at[index, label] = 1

    print(f"Labels adicionadas: {labels}")


def remove_labels(dataframe: pd.DataFrame, index: int, labels: list) -> None:
    invalid_labels = validate_labels(labels)

    if invalid_labels:
        print(f"Labels inválidas: {invalid_labels}")
        return

    for label in labels:
        dataframe.at[index, label] = 0

    print(f"Labels removidas: {labels}")


def set_labels(dataframe: pd.DataFrame, index: int, raw_text: str) -> None:
    assignments = parse_label_list(raw_text)

    for assignment in assignments:
        if "=" not in assignment:
            print(f"Atribuição inválida: {assignment}")
            continue

        label, value = assignment.split("=", 1)

        label = label.strip()
        value = value.strip()

        if label not in SPECTRA_LABELS:
            print(f"Label inválida: {label}")
            continue

        if value not in ["0", "1"]:
            print(f"Valor inválido para {label}: {value}")
            continue

        dataframe.at[index, label] = int(value)

    print("Labels atualizadas.")


def clear_labels(dataframe: pd.DataFrame, index: int) -> None:
    for label in SPECTRA_LABELS:
        dataframe.at[index, label] = 0

    print("Todas as labels foram zeradas.")


def mark_reviewed(dataframe: pd.DataFrame, index: int) -> None:
    dataframe.at[index, "reviewed"] = 1


def mark_drop(dataframe: pd.DataFrame, index: int) -> None:
    dataframe.at[index, "use_frame"] = 0
    dataframe.at[index, "reviewed"] = 1

    print("Frame marcado como descartado.")


def mark_keep(dataframe: pd.DataFrame, index: int) -> None:
    dataframe.at[index, "use_frame"] = 1

    print("Frame marcado como utilizável.")


def save_dataframe(dataframe: pd.DataFrame, output_csv: Path) -> None:
    dataframe.to_csv(output_csv, index=False)


def find_start_index(dataframe: pd.DataFrame, start_index: int, only_unreviewed: bool) -> int:
    if not only_unreviewed:
        return start_index

    for index in range(start_index, len(dataframe)):
        reviewed = int(dataframe.iloc[index].get("reviewed", 0))

        if reviewed == 0:
            return index

    return len(dataframe)


def review_dataset(
    input_csv: Path,
    output_csv: Path,
    scores_csv: Path = None,
    start_index: int = 0,
    only_unreviewed: bool = True,
) -> None:
    dataframe = load_or_create_review_csv(
        input_csv=input_csv,
        output_csv=output_csv,
    )

    validate_label_columns(dataframe)

    scores_dataframe = None

    if scores_csv is not None and scores_csv.exists():
        scores_dataframe = pd.read_csv(scores_csv)

        if len(scores_dataframe) != len(dataframe):
            print(
                "Aviso: CSV de scores tem quantidade de linhas diferente do CSV principal."
            )
            scores_dataframe = None

    index = find_start_index(
        dataframe=dataframe,
        start_index=start_index,
        only_unreviewed=only_unreviewed,
    )

    while index < len(dataframe):
        row = dataframe.iloc[index]
        image_path = Path(row["frame_path"])

        print_frame_info(dataframe, index)
        open_image(image_path)

        while True:
            command = input("\nComando [Enter=próximo | h=ajuda]: ").strip()

            if command == "":
                mark_reviewed(dataframe, index)
                save_dataframe(dataframe, output_csv)
                index += 1
                break

            if command == "h" or command == "help":
                print(HELP_TEXT)
                continue

            if command == "q":
                save_dataframe(dataframe, output_csv)
                print(f"Revisão salva em: {output_csv}")
                return

            if command == "labels":
                print_all_labels()
                continue

            if command == "groups":
                print_groups()
                continue

            if command == "active":
                print_active_labels(dataframe, index)
                continue

            if command == "scores":
                print_scores(scores_dataframe, index)
                continue

            if command == "clear":
                clear_labels(dataframe, index)
                save_dataframe(dataframe, output_csv)
                print_active_labels(dataframe, index)
                continue

            if command == "drop":
                mark_drop(dataframe, index)
                save_dataframe(dataframe, output_csv)
                index += 1
                break

            if command == "keep":
                mark_keep(dataframe, index)
                save_dataframe(dataframe, output_csv)
                continue

            if command == "back":
                save_dataframe(dataframe, output_csv)
                index = max(0, index - 1)
                break

            if command.startswith("a "):
                labels = parse_label_list(command[2:])
                add_labels(dataframe, index, labels)
                save_dataframe(dataframe, output_csv)
                print_active_labels(dataframe, index)
                continue

            if command.startswith("r "):
                labels = parse_label_list(command[2:])
                remove_labels(dataframe, index, labels)
                save_dataframe(dataframe, output_csv)
                print_active_labels(dataframe, index)
                continue

            if command.startswith("set "):
                set_labels(dataframe, index, command[4:])
                save_dataframe(dataframe, output_csv)
                print_active_labels(dataframe, index)
                continue

            print("Comando não reconhecido. Digite 'h' para ajuda.")

    save_dataframe(dataframe, output_csv)

    print("\nRevisão finalizada.")
    print(f"CSV revisado salvo em: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Ferramenta simples para revisar labels da Spectra."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV automático de entrada."
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="CSV revisado de saída."
    )

    parser.add_argument(
        "--scores-csv",
        default=None,
        help="CSV de scores, opcional."
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Índice inicial da revisão."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Revisa todos os frames, inclusive os já revisados."
    )

    args = parser.parse_args()

    scores_csv = Path(args.scores_csv) if args.scores_csv else None

    review_dataset(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        scores_csv=scores_csv,
        start_index=args.start_index,
        only_unreviewed=not args.all,
    )


if __name__ == "__main__":
    main()
    
# python -m tools.review_spectra_dataset \
#   --input-csv data/datasets/spectra_auto_labels.csv \
#   --output-csv data/datasets/spectra_reviewed_labels.csv \
#   --scores-csv data/datasets/spectra_auto_scores.csv
