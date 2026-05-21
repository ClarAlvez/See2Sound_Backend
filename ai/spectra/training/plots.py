from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(history, output_dir):
    """
    Gera gráficos do treinamento.

    Saídas:
    - loss.png
    - f1_score.png
    - precision.png
    - recall.png
    - hamming_accuracy.png
    - exact_match_accuracy.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric(
        history=history,
        train_key="train_loss",
        validation_key="validation_loss",
        title="Training and Validation Loss",
        ylabel="Loss",
        output_path=output_dir / "loss.png",
    )

    _plot_metric(
        history=history,
        train_key="train_f1_score",
        validation_key="validation_f1_score",
        title="Training and Validation F1-score",
        ylabel="F1-score",
        output_path=output_dir / "f1_score.png",
    )

    _plot_metric(
        history=history,
        train_key="train_precision",
        validation_key="validation_precision",
        title="Training and Validation Precision",
        ylabel="Precision",
        output_path=output_dir / "precision.png",
    )

    _plot_metric(
        history=history,
        train_key="train_recall",
        validation_key="validation_recall",
        title="Training and Validation Recall",
        ylabel="Recall",
        output_path=output_dir / "recall.png",
    )

    _plot_metric(
        history=history,
        train_key="train_hamming_accuracy",
        validation_key="validation_hamming_accuracy",
        title="Training and Validation Hamming Accuracy",
        ylabel="Hamming Accuracy",
        output_path=output_dir / "hamming_accuracy.png",
    )

    _plot_metric(
        history=history,
        train_key="train_exact_match_accuracy",
        validation_key="validation_exact_match_accuracy",
        title="Training and Validation Exact Match Accuracy",
        ylabel="Exact Match Accuracy",
        output_path=output_dir / "exact_match_accuracy.png",
    )


def _plot_metric(history, train_key, validation_key, title, ylabel, output_path):
    epochs = range(1, len(history[train_key]) + 1)

    plt.figure()
    plt.plot(epochs, history[train_key], label="Training")
    plt.plot(epochs, history[validation_key], label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()