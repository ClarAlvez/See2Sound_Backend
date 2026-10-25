"""Builder v3 do Spectra Atmosphere.

Reutiliza os leitores de data/external existentes em tools.builder_atmosphere,
mas acrescenta máscaras de supervisão label_mask_<label> para evitar que
labels desconhecidas sejam treinadas como negativas.

Executar da raiz do See2Sound_Backend:
    python -m tools.builder_atmosphere_v3 --external-root data/external
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ai.spectra.Atmosphere.labels import LABELS
from tools import builder_atmosphere as legacy

TIME_LABELS = ("day", "night", "dawn_dusk")
BDD_WEATHER_LABELS = ("clear_weather", "cloudy", "foggy", "rainy", "snowy")
TRANSIENT_LABELS = (
    "day", "night", "dawn_dusk", "bright", "low_light",
    "sunny", "cloudy", "foggy", "rainy", "snowy",
)
MASK_PREFIX = "label_mask_"


def collect_transient_v3(external_root: Path, positive_threshold: float,
                         max_samples: int | None, seed: int):
    annotations_path, image_root, holdout_root = legacy.discover_transient_paths(external_root)
    if annotations_path is None or image_root is None:
        print("AVISO: Transient Attributes não encontrado.")
        return []

    annotations = legacy.load_transient_annotations(annotations_path)
    split_by_path = legacy.load_transient_split(holdout_root)
    dark_index = legacy.TRANSIENT_ATTRIBUTES.index("dark")
    rows = []

    for relative_path, values in annotations.items():
        normalized = relative_path.replace("\\", "/")
        image_path = image_root / Path(normalized)
        if not image_path.exists():
            continue
        row = legacy.transient_values_to_row(
            frame_path=image_path,
            values=values,
            positive_threshold=positive_threshold,
        )
        # Corrige a omissão do builder v2: dark (Transient) -> low_light.
        if "low_light" in LABELS:
            row["low_light"] = int(float(values[dark_index]) >= positive_threshold)
        # Hora do dia é exclusiva: em casos ambíguos, mantém a maior anotação.
        active_times = [label for label in TIME_LABELS if row.get(label, 0) == 1]
        if len(active_times) > 1:
            raw_scores = {
                "day": float(values[legacy.TRANSIENT_ATTRIBUTES.index("daylight")]),
                "night": float(values[legacy.TRANSIENT_ATTRIBUTES.index("night")]),
                "dawn_dusk": max(
                    float(values[legacy.TRANSIENT_ATTRIBUTES.index("sunrisesunset")]),
                    float(values[legacy.TRANSIENT_ATTRIBUTES.index("dawndusk")]),
                ),
            }
            best_time = max(active_times, key=lambda label: raw_scores[label])
            for label in TIME_LABELS:
                row[label] = int(label == best_time)
        if not legacy.active_labels(row):
            continue
        source_split = split_by_path.get(normalized, "")
        rows.append(legacy.finish_row(
            row=row,
            source_dataset="transient_attributes",
            source_category="human_attributes",
            source_split=source_split,
            generated_split=(legacy.normalize_split(source_split) if source_split else None),
        ))

    random.Random(seed).shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def select_bdd_by_time(rows, per_time: int, seed: int):
    """Balanceia BDD por período ANTES de limitar o total.

    Não altera os source_split oficiais da BDD100K.
    """
    buckets = defaultdict(list)
    for row in rows:
        labels = [x for x in TIME_LABELS if float(row.get(x, 0)) >= 0.5]
        if len(labels) == 1:
            buckets[labels[0]].append(row)
    rng = random.Random(seed)
    selected = []
    for label in TIME_LABELS:
        items = buckets[label]
        rng.shuffle(items)
        selected.extend(items[:per_time])
        print(f"BDD100K {label}: {len(items)} disponíveis, {min(len(items), per_time)} selecionados")
    return selected


def add_known_masks(row):
    """1 = existe anotação utilizável para a label; 0 = desconhecida."""
    for label in LABELS:
        row[MASK_PREFIX + label] = 0

    source = row.get("source_dataset", "")
    known = set()
    if source == "bdd100k":
        category = str(row.get("source_category", ""))
        parts = dict(p.split("=", 1) for p in category.split(";") if "=" in p)
        if legacy.BDD_TIME_MAP.get(parts.get("time", "")) in LABELS:
            known.update(TIME_LABELS)
        if legacy.BDD_WEATHER_MAP.get(parts.get("weather", "")) in LABELS:
            known.update(BDD_WEATHER_LABELS)
        # sunny não é anotado separadamente pela BDD100K.
    elif source == "transient_attributes":
        known.update(TRANSIENT_LABELS)
        # Transient não tem anotação equivalente a clear_weather.
    elif source == "sun_attributes":
        known.update(("sunny", "cloudy"))
    elif source == "acdc":
        label = legacy.ACDC_MAP.get(row.get("source_category", ""))
        if label in LABELS:
            known.add(label)
        if label == "night":
            known.update(TIME_LABELS)
    elif source == "exdark":
        known.add("low_light")

    for label in known:
        if label in LABELS:
            row[MASK_PREFIX + label] = 1
    return row


def validate_rows(rows):
    if not rows:
        raise ValueError("Nenhuma amostra válida encontrada. Confira --external-root.")
    for row in rows:
        for label in LABELS:
            if int(row.get(label, 0)) not in (0, 1):
                raise ValueError(f"Label inválida {label}: {row.get(label)}")
            if int(row[MASK_PREFIX + label]) not in (0, 1):
                raise ValueError(f"Máscara inválida: {label}")
            if int(row[label]) == 1 and int(row[MASK_PREFIX + label]) == 0:
                raise ValueError(f"Positivo sem máscara: {label} ({row['frame_path']})")
        if row.get("generated_split") not in ("train", "validation", "test"):
            raise ValueError(f"Split inválido: {row.get('generated_split')}")


def create_report(dataframe):
    result = {
        "total_rows": len(dataframe),
        "sources": dataframe["source_dataset"].value_counts().to_dict(),
        "splits": dataframe["generated_split"].value_counts().to_dict(),
        "per_label": {},
        "by_source": {},
    }
    for label in LABELS:
        known = dataframe[MASK_PREFIX + label].astype(bool)
        positive = dataframe[label].astype(bool) & known
        result["per_label"][label] = {
            "known": int(known.sum()),
            "positive": int(positive.sum()),
            "negative": int((known & ~positive).sum()),
            "unknown": int((~known).sum()),
            "train_positive": int((positive & dataframe["generated_split"].eq("train")).sum()),
            "val_positive": int((positive & dataframe["generated_split"].eq("validation")).sum()),
            "test_positive": int((positive & dataframe["generated_split"].eq("test")).sum()),
        }
    for source, part in dataframe.groupby("source_dataset"):
        result["by_source"][source] = {
            label: {
                "positive": int((part[label].astype(bool) & part[MASK_PREFIX + label].astype(bool)).sum()),
                "known": int(part[MASK_PREFIX + label].sum()),
            }
            for label in LABELS
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Builder Atmosphere v3 com máscaras de supervisão")
    parser.add_argument("--external-root", default="data/external")
    parser.add_argument("--output-csv", default="data/datasets/spectra_atmosphere_v3_masked.csv")
    parser.add_argument("--output-report", default="data/datasets/spectra_atmosphere_v3_masked_report.json")
    parser.add_argument("--positive-threshold", type=float, default=0.70)
    parser.add_argument("--max-acdc-per-condition", type=int, default=800)
    parser.add_argument("--max-transient-samples", type=int, default=4500)
    parser.add_argument("--max-sun-samples", type=int, default=None)
    parser.add_argument("--max-exdark-samples", type=int, default=1200)
    parser.add_argument("--bdd-per-time", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.external_root)
    if not root.exists():
        raise FileNotFoundError(f"data/external não encontrado: {root}")

    rows = []
    loaders = [
        ("ACDC", lambda: legacy.collect_acdc_rows(root, args.max_acdc_per_condition)),
        ("Transient", lambda: collect_transient_v3(root, args.positive_threshold, args.max_transient_samples, args.seed)),
        ("SUN", lambda: legacy.collect_sun_rows(root, args.positive_threshold, args.max_sun_samples)),
        ("ExDark", lambda: legacy.collect_exdark_rows(root, args.max_exdark_samples, args.seed)),
        ("BDD100K", lambda: select_bdd_by_time(legacy.collect_bdd100k_rows(root, max_samples=None), args.bdd_per_time, args.seed)),
    ]
    for name, loader in loaders:
        print(f"\nColetando {name}...")
        new_rows = loader()
        print(f"{name}: {len(new_rows)} amostras")
        rows.extend(new_rows)

    rows = legacy.dedupe_rows(rows)
    rows = [add_known_masks(row) for row in rows]
    random.Random(args.seed).shuffle(rows)
    validate_rows(rows)

    columns = [
        "frame_path", *LABELS,
        *(MASK_PREFIX + label for label in LABELS),
        "primary_label", "generated_split", "source_dataset", "source_category", "source_split",
    ]
    dataframe = pd.DataFrame(rows)
    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = ""
    csv_path = Path(args.output_csv)
    report_path = Path(args.output_report)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe[columns].to_csv(csv_path, index=False)
    report = create_report(dataframe)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("CSV:", csv_path)
    print("Relatório:", report_path)
    print("TOTAL:", len(dataframe))
    print("\nSPLITS:\n", dataframe["generated_split"].value_counts().to_string())
    print("\nFONTES:\n", dataframe["source_dataset"].value_counts().to_string())
    print("\nLABEL                 CONHECIDOS  POSITIVOS  NEGATIVOS  NÃO ANOTADOS")
    for label, counts in report["per_label"].items():
        print(f"{label:<21} {counts['known']:>9} {counts['positive']:>10} {counts['negative']:>10} {counts['unknown']:>13}")
    print("\nATENÇÃO: o treino deve usar ai.spectra.Atmosphere.training_masked")


if __name__ == "__main__":
    main()
