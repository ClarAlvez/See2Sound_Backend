import re
import unicodedata
from typing import Dict, List


class FidelityFilter:
    """
    Filtro simples para detectar possíveis invenções do modelo.

    Ele não é perfeito, mas ajuda a marcar casos em que o LLM gerou detalhes
    que não vieram das labels ou do contexto.
    """

    def __init__(self):
        self.risky_words = {
            # Gênero/idade que o modelo não deve inventar sem label específica.
            "homem": ["man"],
            "mulher": ["woman"],
            "menino": ["boy"],
            "menina": ["girl"],
            "criança": ["child"],

            # Emoções/intenção.
            "assustado": ["scared", "afraid"],
            "assustada": ["scared", "afraid"],
            "feliz": ["happy", "smiling"],
            "triste": ["sad", "crying"],
            "sozinho": ["alone"],
            "sozinha": ["alone"],

            # Características visuais.
            "escura": ["dark"],
            "escuro": ["dark"],
            "iluminada": ["bright"],
            "iluminado": ["bright"],
            "cheia": ["crowded"],
            "cheio": ["crowded"],
            "vazia": ["empty"],
            "vazio": ["empty"],
        }

    def validate(
        self,
        description: str,
        labels: List[str],
        context: Dict,
    ) -> List[str]:
        warnings = []

        normalized_description = self._normalize(description)
        normalized_labels = {
            self._normalize(label)
            for label in labels
            if isinstance(label, str)
        }

        # Também considera labels estruturadas dentro do contexto.
        context_values = self._extract_context_values(context)
        normalized_context_values = {
            self._normalize(value)
            for value in context_values
        }

        available_terms = normalized_labels.union(normalized_context_values)

        for portuguese_word, required_labels in self.risky_words.items():
            normalized_word = self._normalize(portuguese_word)

            if self._contains_word(normalized_description, normalized_word):
                allowed = False

                for required_label in required_labels:
                    if self._normalize(required_label) in available_terms:
                        allowed = True
                        break

                if not allowed:
                    warnings.append(
                        "Possível detalhe inventado na descrição: '{}'.".format(
                            portuguese_word
                        )
                    )

        return warnings

    def _extract_context_values(self, context: Dict) -> List[str]:
        values = []

        if not isinstance(context, dict):
            return values

        for value in context.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        values.append(item)

            elif isinstance(value, dict):
                values.extend(self._extract_context_values(value))

            elif isinstance(value, str):
                values.append(value)

        return values

    def _contains_word(self, text: str, word: str) -> bool:
        pattern = r"\b{}\b".format(re.escape(word))
        return re.search(pattern, text) is not None

    def _normalize(self, text: str) -> str:
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
