"""Treino v3 AtmosphereNet com supervisão parcial (masks por label).

Use somente com o CSV de tools.builder_atmosphere_v3.
O conjunto test não é usado para escolher pesos, hyperparâmetros ou threshold.
"""

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from ai.spectra.Atmosphere.labels import LABELS
from ai.spectra.Atmosphere.model import SpectraAtmosphereNet
from ai.spectra.data.transforms import get_atmosphere_train_transforms, get_validation_transforms

MASK_PREFIX = "label_mask_"


class MaskedAtmosphereDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.frame = dataframe.reset_index(drop=True)
        self.transform = transform
        self.targets = self.frame[LABELS].to_numpy(dtype=np.float32)
        self.masks = self.frame[[MASK_PREFIX + x for x in LABELS]].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = Path(str(self.frame.iloc[index]["frame_path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {path}. Ajuste os caminhos do CSV no ambiente de treino.")
        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))
        return image, torch.from_numpy(self.targets[index]), torch.from_numpy(self.masks[index])


def load_dataframe(path):
    frame = pd.read_csv(path)
    required = ["frame_path", "generated_split", *LABELS, *[MASK_PREFIX + x for x in LABELS]]
    missing = [x for x in required if x not in frame.columns]
    if missing:
        raise ValueError(f"CSV não possui as colunas do Atmosphere v3: {missing}")
    frame["generated_split"] = frame["generated_split"].astype(str).str.lower().str.strip()
    frame.loc[frame["generated_split"].eq("val"), "generated_split"] = "validation"
    invalid = sorted(set(frame["generated_split"]) - {"train", "validation", "test"})
    if invalid:
        raise ValueError(f"generated_split inválido: {invalid}")
    for name in [*LABELS, *[MASK_PREFIX + x for x in LABELS]]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not frame[name].isin([0, 1]).all():
            raise ValueError(f"{name} precisa conter apenas 0 ou 1")
    y = frame[LABELS].to_numpy(dtype=np.float32)
    mask = frame[[MASK_PREFIX + x for x in LABELS]].to_numpy(dtype=np.float32)
    if np.any((y > 0) & (mask == 0)):
        raise ValueError("Há labels positivas cuja máscara é zero; verifique o builder.")
    if not (frame["generated_split"] == "train").any() or not (frame["generated_split"] == "validation").any():
        raise ValueError("O CSV deve conter splits train e validation não vazios")
    if frame["frame_path"].duplicated().any():
        raise ValueError("O CSV contém caminhos de imagem duplicados; deduplique antes de treinar")
    return frame


def compute_pos_weights(frame, cap=6.0, use_pos_weight=True):
    targets = frame[LABELS].to_numpy(dtype=np.float32)
    masks = frame[[MASK_PREFIX + x for x in LABELS]].to_numpy(dtype=np.float32)
    positives = (targets * masks).sum(axis=0)
    known = masks.sum(axis=0)
    negatives = known - positives
    if use_pos_weight:
        weights = np.clip(negatives / np.maximum(positives, 1.0), 1.0, cap)
    else:
        weights = np.ones(len(LABELS), dtype=np.float32)
    print("\nSUPERVISÃO NO TREINO:")
    for name, p, n, k, w in zip(LABELS, positives, negatives, known, weights):
        print(f"{name:<16} positivos={int(p):>5} negativos={int(n):>5} conhecidos={int(k):>5} pos_weight={w:.2f}")
        if p == 0 or n == 0:
            print(f"  AVISO: {name} não tem {('positivos' if p == 0 else 'negativos')} supervisionados suficientes para aprender a distinção.")
    return torch.as_tensor(weights, dtype=torch.float32)


def masked_loss(logits, targets, mask, pos_weights):
    entries = F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=pos_weights,
        reduction="none",
    )
    return (entries * mask).sum() / mask.sum().clamp_min(1.0)


def metrics_from_arrays(y, pred, mask):
    known = mask.astype(bool)
    tp = (pred & y & known).sum(axis=0)
    fp = (pred & ~y & known).sum(axis=0)
    fn = (~pred & y & known).sum(axis=0)
    tn = (~pred & ~y & known).sum(axis=0)

    def prf(t, f_pos, f_neg):
        precision = float(t / max(1, t + f_pos))
        recall = float(t / max(1, t + f_neg))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return precision, recall, f1

    micro_p, micro_r, micro_f1 = prf(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    per_label = {}
    f1s = []
    for i, label in enumerate(LABELS):
        p, r, f1 = prf(int(tp[i]), int(fp[i]), int(fn[i]))
        f1s.append(f1)
        per_label[label] = {
            "precision": p, "recall": r, "f1": f1,
            "known": int(known[:, i].sum()),
            "positives": int((y[:, i] & known[:, i]).sum()),
            "tp": int(tp[i]), "fp": int(fp[i]),
            "fn": int(fn[i]), "tn": int(tn[i]),
        }
    return {
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f1,
        "macro_f1": float(np.mean(f1s)),
        "per_label": per_label,
    }


def run_epoch(model, loader, optimizer, device, pos_weights, threshold=0.5,
              freeze_backbone=False, unfreeze_last_block=False):
    training = optimizer is not None
    if training:
        model.train()
        # BN dos blocos congelados NÃO deve acumular estatísticas novas.
        if freeze_backbone:
            model.backbone.eval()
            if unfreeze_last_block:
                model.backbone.layer4.train()
    else:
        model.eval()

    total_loss = 0.0
    total_known = 0.0
    all_targets, all_preds, all_masks = [], [], []
    with torch.set_grad_enabled(training):
        for images, targets, mask in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = masked_loss(logits, targets, mask, pos_weights)
            if training:
                loss.backward()
                optimizer.step()
            known = float(mask.sum().item())
            total_loss += loss.item() * known
            total_known += known
            if not training:
                all_targets.append(targets.cpu().numpy().astype(bool))
                all_preds.append((torch.sigmoid(logits) >= threshold).cpu().numpy().astype(bool))
                all_masks.append(mask.cpu().numpy().astype(bool))

    avg_loss = total_loss / max(1.0, total_known)
    if training:
        return avg_loss, None
    y = np.concatenate(all_targets)
    pred = np.concatenate(all_preds)
    masks = np.concatenate(all_masks)
    return avg_loss, metrics_from_arrays(y, pred, masks)


def print_metrics(title, metrics):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    print(f"{'LABEL':<16} {'PREC':>8} {'RECALL':>8} {'F1':>8} {'CONHECIDOS':>12} {'POSITIVOS':>11} {'FP':>6} {'FN':>6}")
    for label in LABELS:
        a = metrics["per_label"][label]
        print(f"{label:<16} {a['precision']:>8.4f} {a['recall']:>8.4f} {a['f1']:>8.4f} "
              f"{a['known']:>12} {a['positives']:>11} {a['fp']:>6} {a['fn']:>6}")
    print(f"MICRO P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} | MACRO F1={metrics['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Treino Atmosphere v3 com BCE mascarada")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.35)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-block", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=6.0)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.unfreeze_last_block and not args.freeze_backbone:
        parser.error("Use --freeze-backbone juntamente com --unfreeze-last-block")
    if not (0 < args.threshold < 1):
        parser.error("--threshold precisa estar entre 0 e 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataframe = load_dataframe(args.dataset_path)

    train_ix = dataframe.index[dataframe.generated_split.eq("train")].tolist()
    val_ix = dataframe.index[dataframe.generated_split.eq("validation")].tolist()
    test_ix = dataframe.index[dataframe.generated_split.eq("test")].tolist()
    print("=" * 76)
    print("ATMOSPHERENET V3 - MASKED TRAINING")
    print("Dataset:", args.dataset_path)
    print("GPU:", device, "| backbone:", args.backbone)
    print("Train:", len(train_ix), "| Val:", len(val_ix), "| Test RESERVADO:", len(test_ix))
    print("Freeze:", args.freeze_backbone, "| Layer4 treinável:", args.unfreeze_last_block)
    print("=" * 76)
    if len(test_ix) == 0:
        print("AVISO: não há test no CSV; este treino não terá avaliação final de teste")

    train_full = MaskedAtmosphereDataset(dataframe, get_atmosphere_train_transforms(args.image_size))
    eval_full = MaskedAtmosphereDataset(dataframe, get_validation_transforms(args.image_size))
    train_loader = DataLoader(Subset(train_full, train_ix), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(Subset(eval_full, val_ix), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = (DataLoader(Subset(eval_full, test_ix), batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda")) if test_ix else None)

    pos_weights = compute_pos_weights(dataframe.iloc[train_ix], cap=args.max_pos_weight,
                                     use_pos_weight=not args.no_pos_weight).to(device)
    model = SpectraAtmosphereNet(
        output_size=len(LABELS), image_size=args.image_size,
        dropout_rate=args.dropout_rate, backbone_name=args.backbone,
        pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone,
        unfreeze_last_block=args.unfreeze_last_block,
    ).to(device)
    param_groups = [{"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": args.head_lr}]
    trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if trainable_backbone:
        param_groups.append({"params": trainable_backbone, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_path = out / "atmosphere_net_best.pt"
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, optimizer, device, pos_weights,
                                  threshold=args.threshold, freeze_backbone=args.freeze_backbone,
                                  unfreeze_last_block=args.unfreeze_last_block)
        val_loss, metrics = run_epoch(model, val_loader, None, device, pos_weights, threshold=args.threshold)
        f1 = metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "validation_f1": metrics["f1"], "validation_macro_f1": f1,
                        "precision": metrics["precision"], "recall": metrics["recall"]})
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"micro_f1={metrics['f1']:.4f} macro_f1={f1:.4f}")
        if f1 > best_f1 + 1e-6 or (abs(f1 - best_f1) <= 1e-6 and val_loss < best_loss):
            best_f1, best_loss, best_epoch = f1, val_loss, epoch
            stale = 0
            # Formato de checkpoint compatível com AtmospherePredictor.
            torch.save({
                "model_state_dict": model.state_dict(),
                "labels": list(LABELS),
                "task_name": "atmosphere",
                "config": {
                    "image_size": args.image_size, "dropout_rate": args.dropout_rate,
                    "backbone_name": args.backbone, "pretrained": not args.no_pretrained,
                    "freeze_backbone": args.freeze_backbone,
                    "unfreeze_last_block": args.unfreeze_last_block,
                    "threshold": args.threshold,
                },
                "best_epoch": epoch,
                "best_validation_macro_f1": f1,
                "best_validation_loss": val_loss,
            }, best_path)
            print("  MELHOR checkpoint:", best_path)
        else:
            stale += 1
        if args.patience > 0 and stale >= args.patience:
            print(f"Early stopping: {stale} épocas sem melhora")
            break

    with open(out / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_metrics = run_epoch(model, val_loader, None, device, pos_weights, threshold=args.threshold)
    print_metrics(f"MELHOR EPOCH {best_epoch} - VALIDATION (não é teste)", val_metrics)
    (out / "validation_metrics.json").write_text(
        json.dumps({"loss": val_loss, **val_metrics}, indent=2, ensure_ascii=False), encoding="utf-8")

    if test_loader is not None:
        test_loss, test_metrics = run_epoch(model, test_loader, None, device, pos_weights, threshold=args.threshold)
        print_metrics("TEST FINAL (avaliado uma única vez)", test_metrics)
        (out / "test_metrics.json").write_text(
            json.dumps({"loss": test_loss, **test_metrics}, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        test_metrics = None

    print("\nMelhor época:", best_epoch, "| validation macro F1:", round(best_f1, 4))
    print("Checkpoint:", best_path)
    print("Métricas:", out / "validation_metrics.json", out / "test_metrics.json")


if __name__ == "__main__":
    main()
