from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ai.spectra.Scene.labels import LABELS
from ai.spectra.Scene.model import SpectraSceneNet
from ai.spectra.data.dataset import SpectraImageDataset
from ai.spectra.data.transforms import get_train_transforms, get_validation_transforms


@dataclass
class SceneTrainingConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.3
    validation_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 0
    backbone_name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False


def train_scene_model(dataset_path, output_dir, config=None):
    config = config or SceneTrainingConfig()
    return _train(dataset_path, output_dir, config)


def _train(dataset_path, output_dir, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = SpectraImageDataset(dataset_path, get_train_transforms(config.image_size), label_columns=LABELS)
    validation_data = SpectraImageDataset(dataset_path, get_validation_transforms(config.image_size), label_columns=LABELS)
    validation_size = max(1, int(len(train_data) * config.validation_ratio))
    train_size = len(train_data) - validation_size
    generator = torch.Generator().manual_seed(config.seed)
    train_subset, validation_indices = random_split(range(len(train_data)), [train_size, validation_size], generator=generator)
    train_loader = DataLoader(torch.utils.data.Subset(train_data, train_subset.indices), config.batch_size, shuffle=True, num_workers=config.num_workers)
    validation_loader = DataLoader(torch.utils.data.Subset(validation_data, validation_indices.indices), config.batch_size, num_workers=config.num_workers)
    model = SpectraSceneNet(len(LABELS), config.image_size, config.dropout_rate, config.backbone_name, config.pretrained, config.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "scene_net_best.pt"
    for _ in range(config.epochs):
        _run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = _run_epoch(model, validation_loader, criterion, device)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({"model_state_dict": model.state_dict(), "labels": LABELS, "task_name": "scene", "config": asdict(config)}, checkpoint_path)
    return {"best_model_path": str(checkpoint_path), "best_validation_loss": best_loss}


def _run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None); total = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            if optimizer: optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            if optimizer: loss.backward(); optimizer.step()
            total += loss.item()
    return total / max(1, len(loader))
