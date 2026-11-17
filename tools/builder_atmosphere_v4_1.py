"""Atmosphere v4.1: supervisão parcial, amostragem por fonte/split e auditoria.

Mantém o builder antigo em tools/builder_atmosphere.py para leitura dos datasets.
Não substitui o builder v2/v3 e é compatível com training_masked.py.
Execute a partir da raiz do See2Sound_Backend.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ai.spectra.Atmosphere.labels import LABELS
from tools import builder_atmosphere as legacy

MASK_PREFIX = "label_mask_"
TIME_LABELS = ("day", "night", "dawn_dusk")
BDD_WEATHER_LABELS = ("clear_weather", "cloudy", "foggy", "rainy", "snowy")
ALL_WEATHER_LABELS = ("clear_weather", "sunny", "cloudy", "foggy", "rainy", "snowy")
TRANSIENT_MAP = {
    "day": ("daylight",),
    "night": ("night",),
    "dawn_dusk": ("sunrisesunset", "dawndusk"),
    "bright": ("bright",),
    "low_light": ("dark",),
    "sunny": ("sunny",),
    "cloudy": ("clouds",),
    "foggy": ("fog",),
    "rainy": ("rain",),
    "snowy": ("snow",),
}


def make_row(path, source, category, source_split="", generated_split=None):
    row = legacy.create_empty_row(Path(path))
    row = legacy.finish_row(
        row=row,
        source_dataset=source,
        source_category=category,
        source_split=source_split,
        generated_split=generated_split,
    )
    for label in LABELS:
        row[MASK_PREFIX + label] = 0
    return row


def known(row, label, value):
    if label in LABELS:
        row[label] = int(value)
        row[MASK_PREFIX + label] = 1


def unknown(row, label):
    if label in LABELS:
        row[label] = 0
        row[MASK_PREFIX + label] = 0


def tri_state(row, label, score, positive, negative):
    if score >= positive:
        known(row, label, 1)
    elif score <= negative:
        known(row, label, 0)
    else:
        unknown(row, label)


def finalize(row):
    row["primary_label"] = legacy.infer_primary_label(row)
    return row


def sample_covering(rows, limit, labels, seed, minimum=80):
    """Cobre labels positivas e negativas antes de preencher o limite.

    Não muda a coluna generated_split e não usa imagens de outro split.
    """
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    pool = rows[:]
    rng.shuffle(pool)
    chosen = []
    chosen_ids = set()

    def add_candidates(label, target, desired):
        present = sum(1 for row in chosen if row.get(MASK_PREFIX + label, 0) == 1
                      and row.get(label, 0) == target)
        if present >= desired:
            return
        for row in pool:
            if len(chosen) >= limit or present >= desired:
                break
            ident = id(row)
            if ident in chosen_ids:
                continue
            if row.get(MASK_PREFIX + label, 0) == 1 and row.get(label, 0) == target:
                chosen.append(row)
                chosen_ids.add(ident)
                present += 1

    for label in labels:
        add_candidates(label, 1, minimum)
        add_candidates(label, 0, minimum)

    for row in pool:
        if len(chosen) >= limit:
            break
        if id(row) not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(id(row))
    return chosen


def sample_by_split(rows, max_samples, labels, seed, minimum=80):
    if max_samples is None or len(rows) <= max_samples:
        return rows
    split_buckets = defaultdict(list)
    for row in rows:
        split_buckets[row["generated_split"]].append(row)
    total = len(rows)
    selected = []
    budget = max_samples
    for i, split in enumerate(("validation", "test", "train")):
        pool = split_buckets[split]
        if not pool:
            continue
        # Reserva proporcional por split; a soma final não ultrapassa o máximo.
        if split == "train":
            n = min(len(pool), budget)
        else:
            n = min(len(pool), budget, max(1, round(max_samples * len(pool) / total)))
        sub = sample_covering(pool, n, labels, seed + i * 127, min(minimum, max(1, n // 10)))
        selected.extend(sub)
        budget -= len(sub)
    return selected


def collect_transient(root, positive, negative, max_samples, seed, validation_fraction):
    annotations_path, images_root, holdout_root = legacy.discover_transient_paths(root)
    if not annotations_path or not images_root:
        print("AVISO: Transient não encontrado")
        return []
    attrs = legacy.load_transient_annotations(annotations_path)
    official = legacy.load_transient_split(holdout_root)
    indices = {name: i for i, name in enumerate(legacy.TRANSIENT_ATTRIBUTES)}
    rows = []
    official_matches = 0
    for relative, values in attrs.items():
        normalized = relative.replace("\\", "/")
        image = images_root / Path(normalized)
        if not image.is_file():
            continue
        original_split = official.get(normalized, "")
        if original_split:
            official_matches += 1
        row = make_row(image, "transient_attributes", "human_attributes", original_split,
                       legacy.normalize_split(original_split) if original_split else None)
        scores = {label: max(float(values[indices[key]]) for key in keys)
                  for label, keys in TRANSIENT_MAP.items() if label in LABELS}
        for label, value in scores.items():
            if label not in TIME_LABELS:
                tri_state(row, label, value, positive, negative)
        strong_times = [l for l in TIME_LABELS if scores.get(l, 0) >= positive]
        if len(strong_times) == 1:
            # Horário é exclusivo; um positivo confiável supervisiona os outros dois.
            for label in TIME_LABELS:
                known(row, label, label == strong_times[0])
        else:
            # Vários horários fortes ou nenhum: a cena não tem horário definido.
            for label in TIME_LABELS:
                unknown(row, label)
        if not any(row[MASK_PREFIX + x] for x in LABELS):
            continue
        rows.append(finalize(row))
    print(f"Transient: {official_matches} imagens com split oficial de holdout, "
          f"{len(rows)} imagens com supervisão útil")
    if official and official_matches == 0:
        print("AVISO CRÍTICO: holdout_split encontrado, mas nenhum caminho correspondeu. "
              "Será usado split determinístico por caminho; verifique risco de frames "
              "da mesma câmera em diferentes conjuntos")
    rows = partition_transient_by_webcam(rows, validation_fraction, seed)
    return sample_by_split(rows, max_samples, list(TRANSIENT_MAP), seed, minimum=100)




def partition_transient_by_webcam(rows, fraction, seed, min_supervised=12):
    """Reserva webcams do treino oficial para validação, sem tocar no teste.

    Critério por diretório do frame: Transient/imageAlignedLD/webcam_id/*.jpg.
    """
    if not 0.0 < fraction < 0.5:
        raise ValueError("transient_val_fraction deve estar em (0, 0.5)")
    train = [r for r in rows if r["generated_split"] == "train"]
    holdout = [r for r in rows if r["generated_split"] == "test"]
    unknown_splits = [r for r in rows if r["generated_split"] not in ("train", "test")]
    if unknown_splits:
        raise ValueError("Transient: split não oficial ou inesperado; confira holdout_split")
    if not train or not holdout:
        raise ValueError("Transient: faltam imagens no treino/teste oficial; confira holdout_split")

    cameras = defaultdict(list)
    for row in train:
        # A webcam é o diretório da imagem; o dataset original usa um diretório por câmera.
        cam = Path(row["frame_path"]).parent.name
        cameras[cam].append(row)
    if len(cameras) < 8:
        raise ValueError(
            f"Transient: apenas {len(cameras)} grupos de webcam em train; "
            "não é seguro formar validação por imagem. Verifique diretórios imageAlignedLD."
        )
    rng = random.Random(seed)
    camera_names = sorted(cameras)
    rng.shuffle(camera_names)
    desired = max(1, round(len(train) * fraction))
    selected, val_count = set(), 0
    for cam in camera_names:
        if val_count >= desired:
            break
        # Não esvaziar o conjunto de treino.
        if len(train) - val_count - len(cameras[cam]) < desired:
            continue
        selected.add(cam)
        val_count += len(cameras[cam])

    # Corrige eventuais faltas de supervisão de bright/low_light na validação.
    priority = ("bright", "low_light")
    def counts(lab, value):
        return sum(1 for cam in selected for r in cameras[cam]
                   if r[MASK_PREFIX + lab] == 1 and r[lab] == value)
    for lab in priority:
        for value in (1, 0):
            while counts(lab, value) < min_supervised:
                candidates = [cam for cam in camera_names if cam not in selected
                              and any(r[MASK_PREFIX + lab] and r[lab] == value for r in cameras[cam])]
                if not candidates:
                    raise ValueError(f"Transient: sem webcams suficientes para validar {lab}={value}")
                # Prefere webcams com mais exemplos supervisionados do grupo ausente.
                candidates.sort(key=lambda cam: sum(r[MASK_PREFIX+lab] and r[lab] == value
                                                    for r in cameras[cam]), reverse=True)
                cam = candidates[0]
                selected.add(cam)
                if sum(len(cameras[x]) for x in selected) >= len(train) * .48:
                    raise ValueError("Transient: validação exigiria >48% do treino; reveja min_supervised")

    for cam in selected:
        for row in cameras[cam]:
            row["generated_split"] = "validation"

    remaining = [r for r in rows if r["generated_split"] == "train"]
    validation = [r for r in rows if r["generated_split"] == "validation"]
    if len(remaining) < 1 or not validation:
        raise ValueError("Transient: train ou validation ficou vazio")
    for lab in priority:
        for value in (1, 0):
            left = sum(r[MASK_PREFIX + lab] and r[lab] == value for r in remaining)
            if left == 0:
                raise ValueError(f"Transient: faltou {lab}={value} no treino após dividir webcams")
    print(f"Transient: divisão por webcams = train {len(remaining)}, "
          f"validation {len(validation)}, test {len(holdout)}; "
          f"webcams train/validation = {len(cameras)-len(selected)}/{len(selected)}", flush=True)
    return rows

def collect_sun(root, positive, negative, max_samples, seed):
    label_path, images_mat, attrs_path = legacy.discover_sun_files(root)
    if not label_path or not images_mat or not attrs_path:
        print("AVISO: SUN não encontrado")
        return []
    names = legacy.matlab_string_list(legacy.load_mat_variable(images_mat, ["images"]))
    attrs = legacy.matlab_string_list(legacy.load_mat_variable(attrs_path, ["attributes"]))
    matrix = np.squeeze(np.asarray(legacy.load_mat_variable(
        label_path, ["attributeLabels_continuous", "attribute_labels", "labels"]), dtype=np.float32))
    if matrix.shape[0] != len(names) and matrix.shape[1] == len(names):
        matrix = matrix.T
    if matrix.shape[0] != len(names):
        raise ValueError(f"SUN: {len(names)} nomes e matriz {matrix.shape}")
    images_root = legacy.find_sun_image_root(root, names)
    if images_root is None:
        print("AVISO: SUN não tem imagens acessíveis")
        return []
    col = {
        "sunny": legacy.find_attribute_index(attrs, ["direct sun/sunny", "sunny"]),
        "cloudy": legacy.find_attribute_index(attrs, ["clouds"]),
    }
    if not any(v is not None for v in col.values()):
        raise ValueError("SUN: atributos sunny/clouds não encontrados; não gerar falsas labels")
    rows = []
    for i, name in enumerate(names):
        image = images_root / Path(name.lstrip("/"))
        if not image.is_file():
            continue
        row = make_row(image, "sun_attributes", "human_attributes")
        for label, j in col.items():
            if j is not None:
                tri_state(row, label, float(matrix[i, j]), positive, negative)
        if any(row[MASK_PREFIX + x] for x in LABELS):
            rows.append(finalize(row))
    return sample_by_split(rows, max_samples, ("sunny", "cloudy"), seed + 9, minimum=180)


def collect_acdc(root, max_per_condition, seed):
    # O leitor legado já preserva os splits train/val/test originais.
    # A v4 faz as máscaras explicitamente sem inferir clima desconhecido.
    rows = legacy.collect_acdc_rows(root, max_per_condition=max_per_condition)
    for row in rows:
        for label in LABELS:
            row[MASK_PREFIX + label] = 0
        label = legacy.ACDC_MAP.get(str(row["source_category"]))
        if label in LABELS:
            known(row, label, 1)
        if label == "night":
            for t in TIME_LABELS:
                known(row, t, t == "night")
        finalize(row)
    return rows


def collect_exdark(root, max_samples, seed):
    rows = legacy.collect_exdark_rows(root, max_samples=max_samples, seed=seed)
    for row in rows:
        for label in LABELS:
            row[MASK_PREFIX + label] = 0
        known(row, "low_light", 1)
        finalize(row)
    return rows


def partition_bdd_train(rows, val_fraction, seed):
    """Divide APENAS a train oficial da BDD, estratificando por tempo e clima."""
    groups = defaultdict(list)
    for row in rows:
        source_cat = str(row.get("source_category", ""))
        groups[source_cat].append(row)
    train, validation = [], []
    rng = random.Random(seed)
    for group in groups.values():
        group = group[:]
        rng.shuffle(group)
        n = (min(len(group) - 1, max(1, round(len(group) * val_fraction)))
             if len(group) >= 2 else 0)
        for row in group[:n]:
            row["generated_split"] = "validation"
            validation.append(row)
        train.extend(group[n:])
    return train, validation


def build_bdd_image_index(image_root, split):
    """Indexa imagens BDD recursivamente sem misturar splits oficiais.

    As imagens podem estar em train/<subpasta>/arquivo.jpg. O leitor legado
    procura apenas train/arquivo.jpg e perde tais imagens.
    """
    folder = image_root / split
    if not folder.is_dir():
        raise FileNotFoundError(f"BDD100K: pasta do split inexistente: {folder}")
    lookup = {}
    collisions = []
    files = 0
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in legacy.IMAGE_EXTENSIONS:
            continue
        files += 1
        name = path.name.casefold()
        if name in lookup:
            collisions.append((name, lookup[name], path))
        else:
            lookup[name] = path
    if collisions:
        preview = "\n".join(f"{name}: {a} | {b}" for name, a, b in collisions[:6])
        raise ValueError(
            f"BDD100K {split}: {len(collisions)} nomes duplicados em caminhos diferentes. "
            "Não é seguro vincular anotações pelo nome. Exemplos:\n" + preview
        )
    print(f"BDD índice {split}: {files} arquivos recursivos, {len(lookup)} nomes únicos", flush=True)
    return lookup


def collect_bdd_rows_recursive(root):
    """Lê JSONs oficiais e resolve imagens por nome dentro do split correto."""
    image_root = legacy.find_bdd_images_root(root)
    label_files = legacy.find_bdd_label_files(root)
    if image_root is None or not label_files:
        raise RuntimeError("BDD100K: imagens/JSONs não encontrados em data/external")
    print("BDD imagens:", image_root, flush=True)
    indexes = {
        "train": build_bdd_image_index(image_root, "train"),
        "validation": build_bdd_image_index(image_root, "val"),
    }
    rows = []
    for label_file in label_files:
        source_split = legacy.infer_bdd_split(label_file)
        if source_split not in indexes:
            raise ValueError(f"BDD100K: JSON com split não esperado: {label_file} ({source_split})")
        images = indexes[source_split]
        data = legacy.load_json(label_file)
        if isinstance(data, dict):
            data = data.get("frames") or data.get("images") or data.get("data") or []
        if not isinstance(data, list):
            raise TypeError(f"BDD100K: formato de JSON inválido: {label_file}")
        stats = defaultdict(int)
        missing = []
        for item in data:
            stats["annotations"] += 1
            image_name = item.get("name") or item.get("file_name") or item.get("image")
            attrs = item.get("attributes") or {}
            if not image_name or not isinstance(attrs, dict):
                stats["invalid_record"] += 1
                continue
            time_of_day = legacy.normalize_text(attrs.get("timeofday"))
            weather = legacy.normalize_text(attrs.get("weather"))
            time_label = legacy.BDD_TIME_MAP.get(time_of_day)
            weather_label = legacy.BDD_WEATHER_MAP.get(weather)
            if time_label not in LABELS and weather_label not in LABELS:
                stats["unmapped"] += 1
                continue
            name = str(image_name).replace("\\", "/").rsplit("/", 1)[-1].casefold()
            path = images.get(name)
            if path is None:
                stats["image_missing"] += 1
                if len(missing) < 5:
                    missing.append(name)
                continue
            row = legacy.create_empty_row(path)
            if time_label in LABELS:
                row[time_label] = 1
            if weather_label in LABELS:
                row[weather_label] = 1
            row = legacy.finish_row(
                row=row, source_dataset="bdd100k",
                source_category=f"time={time_of_day};weather={weather}",
                source_split=source_split, generated_split=source_split,
            )
            rows.append(row)
            stats["usable"] += 1
        print(f"BDD JSON {source_split}: {dict(stats)}", flush=True)
        if missing:
            print(f"BDD ausentes {source_split} (exemplos): {missing}", flush=True)
        mapped = stats["annotations"] - stats["unmapped"] - stats["invalid_record"]
        if mapped > 100 and stats["usable"] / mapped < 0.90:
            raise ValueError(
                f"BDD100K {source_split}: somente {stats['usable']}/{mapped} "
                "anotações reconhecidas têm JPG correspondente, mesmo após busca recursiva. "
                "Verifique a pasta e os arquivos JSON; não iniciar treinamento."
            )
    return rows


def collect_bdd(root, seed, max_train, max_val, max_test,
                min_weather_train, min_weather_eval, min_time_train, min_time_eval,
                validation_fraction):
    raw = collect_bdd_rows_recursive(root)
    if not raw:
        print("AVISO: BDD100K sem imagens utilizáveis")
        return []
    originals = defaultdict(list)
    for row in raw:
        source_split = str(row.get("source_split", ""))
        if source_split in ("train", "validation", "test"):
            originals[source_split].append(row)
        else:
            raise ValueError("BDD100K: arquivo de anotações sem split conhecido")
    print("BDD oficiais aceitos:",
          {split: len(group) for split, group in originals.items()}, flush=True)
    if len(originals["validation"]) > 5000 and len(originals["train"]) < 5000:
        raise ValueError(
            "BDD100K: discrepância na fonte oficial: "
            f"apenas {len(originals['train'])} imagens train receberam anotações, "
            f"versus {len(originals['validation'])} na val. "
            "Não é seguro construir o dataset: execute "
            "python -m tools.audit_bdd_atmosphere --external-root data/external "
            "e verifique os nomes das imagens/JSONs."
        )
    train, validation = partition_bdd_train(originals["train"], validation_fraction, seed)
    test = originals["validation"] + originals["test"]
    if not test:
        print("AVISO: BDD100K sem val/test oficial; reservando teste a partir da train")
        train, test = partition_bdd_train(train, 0.15, seed + 55)
        for row in test:
            row["generated_split"] = "test"
    else:
        # A validação OFICIAL da BDD fica como holdout TEST, nunca usada no treinamento.
        for row in test:
            row["generated_split"] = "test"
    for split_name, pool in (("train", train), ("validation", validation), ("test", test)):
        print(f"BDD bruto {split_name}: {len(pool)}")

    def choose(pool, cap, time_min, weather_min, local_seed):
        if cap is None or len(pool) <= cap:
            return pool
        rng = random.Random(local_seed)
        shuffled = pool[:]
        rng.shuffle(shuffled)
        output, used = [], set()

        def add_label(label, quota):
            have = sum(1 for r in output if r.get(label, 0) == 1)
            for r in shuffled:
                if len(output) >= cap or have >= quota:
                    break
                if r.get(label, 0) != 1 or id(r) in used:
                    continue
                output.append(r)
                used.add(id(r))
                have += 1

        # Tempo limpo e condições raras são priorizados ANTES de preencher aleatoriamente.
        for label in BDD_WEATHER_LABELS:
            add_label(label, weather_min)
        for label in TIME_LABELS:
            add_label(label, time_min)
        for r in shuffled:
            if len(output) >= cap:
                break
            if id(r) not in used:
                output.append(r)
                used.add(id(r))
        return output

    selected = []
    for n, (part, cap, tmin, wmin) in enumerate((
        (train, max_train, min_time_train, min_weather_train),
        (validation, max_val, min_time_eval, min_weather_eval),
        (test, max_test, min_time_eval, min_weather_eval),
    )):
        selected.extend(choose(part, cap, tmin, wmin, seed + 97 * n))

    for row in selected:
        for label in LABELS:
            row[MASK_PREFIX + label] = 0
        category = str(row["source_category"])
        parts = dict(part.split("=", 1) for part in category.split(";") if "=" in part)
        time_label = legacy.BDD_TIME_MAP.get(parts.get("time", ""))
        weather_label = legacy.BDD_WEATHER_MAP.get(parts.get("weather", ""))
        if time_label in LABELS:
            for t in TIME_LABELS:
                known(row, t, t == time_label)
        if weather_label in LABELS:
            # Só o campo weather do BDD é supervisionado como exclusivo ENTRE
            # as condições BDD. sunny não é um atributo nativo da BDD.
            for w in BDD_WEATHER_LABELS:
                # Tempo chuvoso/nebuloso/com neve pode continuar nublado.
                # BDD anota UMA condição dominante; não é uma prova
                # visual de ausência de nuvens nesses casos.
                if w == "cloudy" and weather_label in ("rainy", "snowy", "foggy"):
                    unknown(row, w)
                else:
                    known(row, w, w == weather_label)
        finalize(row)
    return selected


def safe_dedupe(rows):
    unique = {}
    blocked = set()
    duplicates = 0
    conflicts = 0
    for row in rows:
        key = str(Path(row["frame_path"]).resolve()).casefold()
        if key not in unique:
            unique[key] = row
            continue
        duplicates += 1
        prev = unique[key]
        if prev["generated_split"] != row["generated_split"]:
            raise ValueError("Vazamento de split: imagem aparece em dois conjuntos: " + key)
        for label in LABELS:
            m = MASK_PREFIX + label
            if (key, label) in blocked:
                continue
            if not prev[m] and row[m]:
                prev[m], prev[label] = 1, row[label]
            elif prev[m] and row[m] and prev[label] != row[label]:
                prev[m], prev[label] = 0, 0
                blocked.add((key, label))
                conflicts += 1
        finalize(prev)
    if duplicates:
        print(f"Dedupe: {duplicates} caminhos repetidos; {conflicts} conflitos de anotação mascarados")
    return list(unique.values())


def make_report(frame, positive, negative):
    report = {
        "total_rows": int(len(frame)),
        "thresholds": {"positive": positive, "negative": negative},
        "splits": frame["generated_split"].value_counts().to_dict(),
        "sources": frame["source_dataset"].value_counts().to_dict(),
        "per_label": {},
        "by_source": {},
    }
    for label in LABELS:
        mask_col = MASK_PREFIX + label
        totals = {}
        for split in ("train", "validation", "test"):
            part = frame.loc[frame.generated_split == split]
            knowns = part[mask_col].eq(1)
            pos = part[label].eq(1) & knowns
            totals[split] = {
                "known": int(knowns.sum()), "positive": int(pos.sum()),
                "negative": int((knowns & ~pos).sum()),
                "unknown": int((~knowns).sum()),
            }
        report["per_label"][label] = totals
    for source, part in frame.groupby("source_dataset"):
        report["by_source"][source] = {}
        for label in LABELS:
            knowns = part[MASK_PREFIX + label].eq(1)
            report["by_source"][source][label] = {
                "known": int(knowns.sum()),
                "positive": int((knowns & part[label].eq(1)).sum()),
                "negative": int((knowns & part[label].eq(0)).sum()),
            }
    return report


def validate(frame, require_clear_eval=True):
    if frame.empty:
        raise ValueError("Nenhuma amostra encontrada")
    for name in ("frame_path", "generated_split", "source_dataset", *LABELS,
                 *(MASK_PREFIX + x for x in LABELS)):
        if name not in frame.columns:
            raise ValueError("Coluna ausente: " + name)
    for label in LABELS:
        m = MASK_PREFIX + label
        if not frame[label].isin((0, 1)).all() or not frame[m].isin((0, 1)).all():
            raise ValueError("Valor inválido: " + label)
        if ((frame[label] == 1) & (frame[m] == 0)).any():
            raise ValueError("Label positiva com máscara zero: " + label)
    if frame.frame_path.duplicated().any():
        raise ValueError("Imagem repetida no CSV")
    if not frame.generated_split.isin(("train", "validation", "test")).all():
        raise ValueError("Split inválido")
    for split in ("train", "validation", "test"):
        if not frame.generated_split.eq(split).any():
            raise ValueError(f"Split vazio: {split}")
    # A maioria das labels deve ter positivos e negativos em todos os splits.
    warnings = []
    for label in LABELS:
        for split in ("train", "validation", "test"):
            part = frame.loc[frame.generated_split.eq(split)]
            m = part[MASK_PREFIX + label].eq(1)
            p = int((m & part[label].eq(1)).sum())
            n = int((m & part[label].eq(0)).sum())
            if p == 0 or n == 0:
                warnings.append(f"{label}/{split}: positivos={p}, negativos={n}")
    for line in warnings:
        print("AVISO cobertura:", line)
    for label in ("bright", "low_light"):
        part = frame.loc[frame.generated_split.eq("validation")]
        m = part[MASK_PREFIX + label].eq(1)
        p = int((m & part[label].eq(1)).sum())
        n = int((m & part[label].eq(0)).sum())
        if p == 0 or n == 0:
            raise ValueError(f"Sem positivos/negativos em validação: {label} (+{p}/-{n})")
    for split in ("validation", "test"):
        part = frame.loc[frame.generated_split.eq(split)]
        m = part[MASK_PREFIX + "clear_weather"].eq(1)
        p = int((m & part["clear_weather"].eq(1)).sum())
        n = int((m & part["clear_weather"].eq(0)).sum())
        if require_clear_eval and (p == 0 or n == 0):
            raise ValueError(f"clear_weather sem cobertura {split} (positivo={p}, negativo={n}); "
                             "verifique anotação BDD val/test ou use --allow-incomplete-clear-eval somente para diagnóstico")
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Atmosphere v4.1: builder multilabel com máscaras, clima estratificado e auditoria")
    parser.add_argument("--external-root", default="data/external")
    parser.add_argument("--output-csv", default="data/datasets/spectra_atmosphere_v4_masked.csv")
    parser.add_argument("--output-report", default="data/datasets/spectra_atmosphere_v4_masked_report.json")
    parser.add_argument("--positive-threshold", type=float, default=0.70)
    parser.add_argument("--negative-threshold", type=float, default=0.30)
    parser.add_argument("--max-acdc-per-condition", type=int, default=600)
    parser.add_argument("--max-transient-samples", type=int, default=6000)
    parser.add_argument("--transient-val-fraction", type=float, default=0.15)
    parser.add_argument("--max-sun-samples", type=int, default=7500)
    parser.add_argument("--max-exdark-samples", type=int, default=450)
    parser.add_argument("--bdd-max-train", type=int, default=8500)
    parser.add_argument("--bdd-max-validation", type=int, default=1600)
    parser.add_argument("--bdd-max-test", type=int, default=1800)
    parser.add_argument("--bdd-min-weather-train", type=int, default=450)
    parser.add_argument("--bdd-min-weather-eval", type=int, default=100)
    parser.add_argument("--bdd-min-time-train", type=int, default=900)
    parser.add_argument("--bdd-min-time-eval", type=int, default=150)
    parser.add_argument("--bdd-val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete-clear-eval", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.negative_threshold < args.positive_threshold <= 1:
        parser.error("Use 0 <= --negative-threshold < --positive-threshold <= 1")
    root = Path(args.external_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    source_calls = (
        ("ACDC", lambda: collect_acdc(root, args.max_acdc_per_condition, args.seed)),
        ("Transient", lambda: collect_transient(root, args.positive_threshold, args.negative_threshold,
                                                args.max_transient_samples, args.seed,
                                                args.transient_val_fraction)),
        ("SUN", lambda: collect_sun(root, args.positive_threshold, args.negative_threshold,
                                    args.max_sun_samples, args.seed)),
        ("ExDark", lambda: collect_exdark(root, args.max_exdark_samples, args.seed)),
        ("BDD100K", lambda: collect_bdd(root, args.seed,
                                       args.bdd_max_train, args.bdd_max_validation, args.bdd_max_test,
                                       args.bdd_min_weather_train, args.bdd_min_weather_eval,
                                       args.bdd_min_time_train, args.bdd_min_time_eval,
                                       args.bdd_val_fraction)),
    )
    rows = []
    for name, load in source_calls:
        print(f"\nColetando {name}...", flush=True)
        chunk = load()
        print(f"{name}: {len(chunk)} imagens", flush=True)
        rows.extend(chunk)
    rows = safe_dedupe(rows)
    random.Random(args.seed).shuffle(rows)
    columns = ["frame_path", *LABELS, *(MASK_PREFIX + x for x in LABELS),
               "primary_label", "generated_split", "source_dataset", "source_category", "source_split"]
    frame = pd.DataFrame(rows, columns=columns)
    report = make_report(frame, args.positive_threshold, args.negative_threshold)
    report["warnings"] = validate(frame, require_clear_eval=not args.allow_incomplete_clear_eval)
    output_csv = Path(args.output_csv)
    output_report = Path(args.output_report)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nTOTAL:", len(frame), "| Splits:", report["splits"], flush=True)
    for label, parts in report["per_label"].items():
        print(f"{label:<16}", " ".join(f"{s}: +{parts[s]['positive']}/-{parts[s]['negative']}"
                                     for s in ("train", "validation", "test")))
    print("CSV:", output_csv)
    print("Relatório:", output_report)
    print("Use o treinamento: python -m ai.spectra.Atmosphere.training_masked")


if __name__ == "__main__":
    main()
