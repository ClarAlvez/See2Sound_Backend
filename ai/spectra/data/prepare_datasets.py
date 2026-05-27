import json
from pathlib import Path

import pandas as pd

from ai.spectra.labels.label_sets import get_labels_for_task


def create_empty_dataset_template(output_csv_path, task_name):
    """
    Cria um CSV vazio no formato esperado por SpectraImageDataset.

    Use para iniciar datasets manuais de:
    - person: crops de pessoas
    - object: crops/regiões/frames com objetos relevantes
    """
    labels = get_labels_for_task(task_name)
    dataframe = pd.DataFrame(columns=["frame_path"] + labels)

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv_path, index=False)

    return output_csv_path


def create_person_dataset_from_manifest(
    manifest_csv_path,
    output_csv_path,
    image_column="frame_path",
    labels_column="labels",
    label_separator=";",
):
    """
    Converte um manifesto de crops de pessoas para o formato da Spectra.

    O manifesto pode ter:
    - uma coluna com o caminho do crop
    - uma coluna "labels" com labels separadas por ponto e vírgula

    Exemplo:
        frame_path,labels
        crops/person_001.jpg,person;black_hair;glasses;blue_clothes
    """
    return _create_multilabel_dataset_from_manifest(
        manifest_csv_path=manifest_csv_path,
        output_csv_path=output_csv_path,
        task_name="person",
        image_column=image_column,
        labels_column=labels_column,
        label_separator=label_separator,
    )


def create_object_dataset_from_manifest(
    manifest_csv_path,
    output_csv_path,
    image_column="frame_path",
    labels_column="labels",
    label_separator=";",
):
    """
    Converte um manifesto de objetos para o formato da Spectra.

    Pode apontar para crops de objetos, regiões candidatas ou frames inteiros.
    """
    return _create_multilabel_dataset_from_manifest(
        manifest_csv_path=manifest_csv_path,
        output_csv_path=output_csv_path,
        task_name="object",
        image_column=image_column,
        labels_column=labels_column,
        label_separator=label_separator,
    )


def create_object_dataset_from_coco_annotations(
    annotations_json_path,
    output_csv_path,
    image_root_dir=None,
    category_mapping=None,
):
    """
    Cria um CSV multilabel de objetos a partir de annotations COCO.

    Não baixa nem recorta imagens. Apenas agrega labels por imagem.
    category_mapping pode mapear categorias COCO para labels da Spectra:

        {"cell phone": "phone", "tv": "television"}
    """
    labels = get_labels_for_task("object")
    label_set = set(labels)
    category_mapping = category_mapping or {}

    with open(annotations_json_path, "r", encoding="utf-8") as file:
        coco_data = json.load(file)

    image_by_id = {
        image["id"]: image["file_name"]
        for image in coco_data.get("images", [])
    }

    category_by_id = {
        category["id"]: category["name"]
        for category in coco_data.get("categories", [])
    }

    labels_by_image_id = {
        image_id: set()
        for image_id in image_by_id
    }

    for annotation in coco_data.get("annotations", []):
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        category_name = category_by_id.get(category_id)

        label = _map_external_label(
            external_label=category_name,
            label_set=label_set,
            label_mapping=category_mapping,
        )

        if label is not None and image_id in labels_by_image_id:
            labels_by_image_id[image_id].add(label)

    rows = []

    for image_id, file_name in image_by_id.items():
        frame_path = _join_optional_root(image_root_dir, file_name)
        rows.append(_build_multilabel_row(frame_path, labels, labels_by_image_id[image_id]))

    return _save_rows(rows, output_csv_path)


def create_object_dataset_from_open_images_annotations(
    annotations_csv_path,
    class_descriptions_csv_path,
    output_csv_path,
    image_root_dir=None,
    label_mapping=None,
):
    """
    Cria um CSV multilabel de objetos a partir de annotations do Open Images.

    Espera arquivos no formato comum:
    - annotations com ImageID e LabelName
    - class_descriptions com LabelName e DisplayName

    Não baixa imagens nem aplica crops. O frame_path fica como ImageID.jpg
    quando image_root_dir é informado, ou apenas ImageID.jpg caso contrário.
    """
    labels = get_labels_for_task("object")
    label_set = set(labels)
    label_mapping = label_mapping or {}

    annotations = pd.read_csv(annotations_csv_path)
    descriptions = pd.read_csv(
        class_descriptions_csv_path,
        header=None,
        names=["LabelName", "DisplayName"],
    )

    display_name_by_label = dict(
        zip(descriptions["LabelName"], descriptions["DisplayName"])
    )

    labels_by_image_id = {}

    for _, row in annotations.iterrows():
        image_id = row["ImageID"]
        label_name = row["LabelName"]
        display_name = display_name_by_label.get(label_name)

        label = _map_external_label(
            external_label=display_name,
            label_set=label_set,
            label_mapping=label_mapping,
        )

        if label is None:
            continue

        labels_by_image_id.setdefault(image_id, set()).add(label)

    rows = []

    for image_id, active_labels in labels_by_image_id.items():
        frame_path = _join_optional_root(image_root_dir, "{}.jpg".format(image_id))
        rows.append(_build_multilabel_row(frame_path, labels, active_labels))

    return _save_rows(rows, output_csv_path)


def _create_multilabel_dataset_from_manifest(
    manifest_csv_path,
    output_csv_path,
    task_name,
    image_column,
    labels_column,
    label_separator,
):
    labels = get_labels_for_task(task_name)
    label_set = set(labels)
    manifest = pd.read_csv(manifest_csv_path)

    _validate_manifest_columns(
        manifest=manifest,
        required_columns=[image_column, labels_column],
    )

    rows = []

    for _, row in manifest.iterrows():
        active_labels = _parse_label_cell(row[labels_column], label_separator)
        unknown_labels = sorted(active_labels - label_set)

        if unknown_labels:
            raise ValueError(
                "Labels desconhecidas para task {}: {}".format(
                    task_name,
                    unknown_labels,
                )
            )

        rows.append(
            _build_multilabel_row(
                frame_path=row[image_column],
                labels=labels,
                active_labels=active_labels,
            )
        )

    return _save_rows(rows, output_csv_path)


def _validate_manifest_columns(manifest, required_columns):
    missing_columns = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas ausentes no manifesto: {}".format(missing_columns)
        )


def _parse_label_cell(value, label_separator):
    if pd.isna(value):
        return set()

    return {
        label.strip()
        for label in str(value).split(label_separator)
        if label.strip()
    }


def _build_multilabel_row(frame_path, labels, active_labels):
    row = {"frame_path": str(frame_path)}

    for label in labels:
        row[label] = 1 if label in active_labels else 0

    return row


def _map_external_label(external_label, label_set, label_mapping):
    if external_label is None:
        return None

    normalized_label = str(external_label).strip().lower().replace(" ", "_")
    mapped_label = label_mapping.get(external_label, label_mapping.get(normalized_label))

    if mapped_label is None:
        mapped_label = normalized_label

    if mapped_label not in label_set:
        return None

    return mapped_label


def _join_optional_root(image_root_dir, image_path):
    if image_root_dir is None:
        return image_path

    return Path(image_root_dir) / image_path


def _save_rows(rows, output_csv_path):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_csv_path, index=False)

    return output_csv_path
