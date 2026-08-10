from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai.spectra.Correlation.labels import LABELS
from ai.spectra.Correlation.model import SpectraCorrelationNet


@dataclass
class CorrelationTrainingConfig:
    input_size: int
    hidden_size: int = 128
    num_layers: int = 1
    dropout_rate: float = 0.3
    bidirectional: bool = True
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 3e-4


def train_correlation_model(sequences, targets, output_dir, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = TensorDataset(torch.as_tensor(sequences, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = SpectraCorrelationNet(config.input_size, config.hidden_size, len(LABELS), config.num_layers, config.dropout_rate, config.bidirectional).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate); criterion = nn.BCEWithLogitsLoss()
    for _ in range(config.epochs):
        model.train()
        for sequence, target in loader:
            optimizer.zero_grad(); loss = criterion(model(sequence.to(device)), target.to(device)); loss.backward(); optimizer.step()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); checkpoint_path = output_dir / "correlation_net.pt"
    torch.save({"model_state_dict": model.state_dict(), "labels": LABELS, "task_name": "correlation", "config": asdict(config)}, checkpoint_path)
    return {"model_path": str(checkpoint_path)}
