import json
from pathlib import Path


class TrainingHistory:
    """
    Guarda o histórico do treinamento ao longo das épocas.
    """

    def __init__(self):
        self.history = {
            "train_loss": [],
            "validation_loss": [],

            "train_precision": [],
            "validation_precision": [],

            "train_recall": [],
            "validation_recall": [],

            "train_f1_score": [],
            "validation_f1_score": [],

            "train_hamming_accuracy": [],
            "validation_hamming_accuracy": [],

            "train_exact_match_accuracy": [],
            "validation_exact_match_accuracy": [],
        }

    def add_epoch(self, train_result, validation_result):
        self.history["train_loss"].append(train_result["loss"])
        self.history["validation_loss"].append(validation_result["loss"])

        self.history["train_precision"].append(train_result["precision"])
        self.history["validation_precision"].append(validation_result["precision"])

        self.history["train_recall"].append(train_result["recall"])
        self.history["validation_recall"].append(validation_result["recall"])

        self.history["train_f1_score"].append(train_result["f1_score"])
        self.history["validation_f1_score"].append(validation_result["f1_score"])

        self.history["train_hamming_accuracy"].append(train_result["hamming_accuracy"])
        self.history["validation_hamming_accuracy"].append(validation_result["hamming_accuracy"])

        self.history["train_exact_match_accuracy"].append(train_result["exact_match_accuracy"])
        self.history["validation_exact_match_accuracy"].append(validation_result["exact_match_accuracy"])

    def to_dict(self):
        return self.history

    def save_json(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4, ensure_ascii=False)