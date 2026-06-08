import re
import unicodedata
from typing import List, Set


class RedundancyFilter:
    """
    Reduz repetições entre cenas próximas.

    Usa:
    1. comparação entre labels;
    2. comparação entre textos gerados.
    """

    def __init__(
        self,
        label_similarity_threshold: float = 0.75,
        text_similarity_threshold: float = 0.85,
    ):
        self.label_similarity_threshold = label_similarity_threshold
        self.text_similarity_threshold = text_similarity_threshold

    def is_too_similar_by_labels(
        self,
        previous_labels: List[str],
        current_labels: List[str],
    ) -> bool:
        previous_set = self._labels_to_set(previous_labels)
        current_set = self._labels_to_set(current_labels)

        if not previous_set or not current_set:
            return False

        similarity = self._jaccard_similarity(previous_set, current_set)

        return similarity >= self.label_similarity_threshold

    def is_too_similar_by_text(
        self,
        previous_description: str,
        current_description: str,
    ) -> bool:
        previous_set = self._text_to_set(previous_description)
        current_set = self._text_to_set(current_description)

        if not previous_set or not current_set:
            return False

        similarity = self._jaccard_similarity(previous_set, current_set)

        return similarity >= self.text_similarity_threshold

    def is_exact_same_text(
        self,
        previous_description: str,
        current_description: str,
    ) -> bool:
        previous = self._normalize_text(previous_description)
        current = self._normalize_text(current_description)

        if not previous or not current:
            return False

        return previous == current

    def _labels_to_set(self, labels: List[str]) -> Set[str]:
        result = set()

        for label in labels:
            if label and isinstance(label, str):
                normalized = self._normalize_text(label)
                if normalized:
                    result.add(normalized)

        return result

    def _text_to_set(self, text: str) -> Set[str]:
        normalized = self._normalize_text(text)

        if not normalized:
            return set()

        words = normalized.split()

        stopwords = {
            "a",
            "o",
            "as",
            "os",
            "um",
            "uma",
            "uns",
            "umas",
            "de",
            "da",
            "do",
            "das",
            "dos",
            "em",
            "na",
            "no",
            "nas",
            "nos",
            "pela",
            "pelo",
            "pelas",
            "pelos",
            "para",
            "por",
            "com",
            "e",
            "ao",
            "a",
            "durante",
            "pela",
            "pelo",
        }

        return {
            word
            for word in words
            if word not in stopwords and len(word) > 2
        }

    def _jaccard_similarity(
        self,
        first_set: Set[str],
        second_set: Set[str],
    ) -> float:
        intersection = first_set.intersection(second_set)
        union = first_set.union(second_set)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip().lower()

        text = unicodedata.normalize("NFD", text)
        text = "".join(
            char
            for char in text
            if unicodedata.category(char) != "Mn"
        )

        text = re.sub(r"[^a-zA-Z0-9À-ÿ\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()
