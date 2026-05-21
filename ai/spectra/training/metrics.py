import torch


def calculate_multilabel_metrics(logits, labels, threshold=0.5):
    """
    Calcula métricas para classificação multilabel.

    logits:
        saída crua da rede, sem sigmoid.

    labels:
        tensor com 0 e 1.

    threshold:
        limite para transformar probabilidade em classe prevista.
    """
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    labels = labels.float()

    true_positive = (predictions * labels).sum().item()
    false_positive = (predictions * (1 - labels)).sum().item()
    false_negative = ((1 - predictions) * labels).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)

    f1_score = (
        2 * precision * recall / (precision + recall + 1e-8)
    )

    hamming_accuracy = (
        (predictions == labels).float().mean().item()
    )

    exact_match_accuracy = (
        (predictions == labels).all(dim=1).float().mean().item()
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "hamming_accuracy": hamming_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
    }


def calculate_loss_average(total_loss, data_loader):
    """
    Evita divisão por zero caso o DataLoader esteja vazio.
    """
    if len(data_loader) == 0:
        return 0.0

    return total_loss / len(data_loader)