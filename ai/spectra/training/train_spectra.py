from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ai.spectra.dataset import SpectraDataset
from ai.spectra.feature_extractor import SpectraFeatureExtractor
from ai.spectra.label_sets import SPECTRA_LABELS
from ai.spectra.network import SpectraNet


DATASET_PATH = Path("data/datasets/spectra_auto_labels.csv")
MODEL_OUTPUT_PATH = Path("data/models/spectra_net.pt")


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = SpectraFeatureExtractor(device=device)

    dataset = SpectraDataset(
        csv_path=DATASET_PATH,
        feature_extractor=feature_extractor
    )

    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False
    )

    input_size = 512
    output_size = len(SPECTRA_LABELS)

    model = SpectraNet(
        input_size=input_size,
        output_size=output_size
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        validation_loss = evaluate(
            model=model,
            validation_loader=validation_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f}"
        )

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": SPECTRA_LABELS,
            "input_size": input_size,
            "output_size": output_size,
        },
        MODEL_OUTPUT_PATH
    )

    print(f"Modelo da Spectra salvo em: {MODEL_OUTPUT_PATH}")


def evaluate(model, validation_loader, criterion, device):
    if len(validation_loader) == 0:
        return 0.0

    model.eval()
    total_validation_loss = 0.0

    with torch.no_grad():
        for features, labels in validation_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_validation_loss += loss.item()

    return total_validation_loss / len(validation_loader)


if __name__ == "__main__":
    train()