import re
import unicodedata
from typing import Dict, List


class FidelityFilter:
    """
    Filtro para detectar possíveis invenções e contradições
    nas descrições geradas pelo modelo.

    O objetivo é verificar se o texto permanece coerente
    com as labels e com o contexto detectado pela Spectra.
    """

    def __init__(self):
        self.risky_words = {
            # Gênero/idade que o modelo não deve inventar
            # sem label específica.
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

        available_terms = normalized_labels.union(
            normalized_context_values
        )

        # ---------------------------------------------------------
        # Detecta possíveis informações inventadas
        # ---------------------------------------------------------

        for portuguese_word, required_labels in self.risky_words.items():
            normalized_word = self._normalize(portuguese_word)

            if self._contains_word(
                normalized_description,
                normalized_word,
            ):
                allowed = False

                for required_label in required_labels:
                    if self._normalize(required_label) in available_terms:
                        allowed = True
                        break

                if not allowed:
                    warnings.append(
                        "Possível detalhe inventado na descrição: '{}'."
                        .format(portuguese_word)
                    )

        # ---------------------------------------------------------
        # Contradições de gênero
        # ---------------------------------------------------------

        if "man" in normalized_labels:
            feminine_terms = [
                "ela",
                "mulher",
                "menina",
            ]

            for term in feminine_terms:
                if self._contains_word(
                    normalized_description,
                    term,
                ):
                    warnings.append(
                        "Contradição de gênero: "
                        "label 'man', mas a descrição utiliza '{}'."
                        .format(term)
                    )

        if "woman" in normalized_labels:
            masculine_terms = [
                "ele",
                "homem",
                "menino",
            ]

            for term in masculine_terms:
                if self._contains_word(
                    normalized_description,
                    term,
                ):
                    warnings.append(
                        "Contradição de gênero: "
                        "label 'woman', mas a descrição utiliza '{}'."
                        .format(term)
                    )

        if "boy" in normalized_labels:
            feminine_terms = [
                "ela",
                "mulher",
                "menina",
            ]

            for term in feminine_terms:
                if self._contains_word(
                    normalized_description,
                    term,
                ):
                    warnings.append(
                        "Contradição de gênero: "
                        "label 'boy', mas a descrição utiliza '{}'."
                        .format(term)
                    )

        if "girl" in normalized_labels:
            masculine_terms = [
                "ele",
                "homem",
                "menino",
            ]

            for term in masculine_terms:
                if self._contains_word(
                    normalized_description,
                    term,
                ):
                    warnings.append(
                        "Contradição de gênero: "
                        "label 'girl', mas a descrição utiliza '{}'."
                        .format(term)
                    )

        # ---------------------------------------------------------
        # Contradições de ação
        # ---------------------------------------------------------

        if "running" in normalized_labels:
            walking_terms = [
                "caminha",
                "caminhando",
                "anda",
                "andando",
            ]

            for term in walking_terms:
                if self._contains_word(
                    normalized_description,
                    term,
                ):
                    warnings.append(
                        "Contradição de ação: "
                        "label 'running', mas a descrição utiliza '{}'."
                        .format(term)
                    )

        if "walking" in normalized_labels:
            running_terms = [
                "corre",
                "correndo",
            ]

            for term in running_terms:
                if self._contains_word(
                    normalized_description,
                    term,
                ):
                    warnings.append(
                        "Contradição de ação: "
                        "label 'walking', mas a descrição utiliza '{}'."
                        .format(term)
                    )

        if "sitting" in normalized_labels:
            standing_terms = [
                "em pé",
                "de pé",
            ]

            for term in standing_terms:
                if self._contains_phrase(
                    normalized_description,
                    term,
                ):
                    warnings.append(
                        "Contradição de ação: "
                        "label 'sitting', mas a descrição utiliza '{}'."
                        .format(term)
                    )

        if "standing" in normalized_labels:
            sitting_terms = [
                "sentado",
                "sentada",
                "sentados",
                "sentadas",
            ]

            for term in sitting_terms:
                if self._contains_word(
                    normalized_description,
                    term,
                ):
                    warnings.append(
                        "Contradição de ação: "
                        "label 'standing', mas a descrição utiliza '{}'."
                        .format(term)
                    )

        # ---------------------------------------------------------
        # Verifica algumas informações importantes omitidas
        # ---------------------------------------------------------

        if "man" in normalized_labels:
            gender_terms = [
                "homem",
                "ele",
            ]

            if not self._contains_any_word(
                normalized_description,
                gender_terms,
            ):
                warnings.append(
                    "Informação possivelmente omitida: "
                    "a label 'man' está presente, mas o gênero masculino "
                    "não aparece claramente na descrição."
                )

        if "woman" in normalized_labels:
            gender_terms = [
                "mulher",
                "ela",
            ]

            if not self._contains_any_word(
                normalized_description,
                gender_terms,
            ):
                warnings.append(
                    "Informação possivelmente omitida: "
                    "a label 'woman' está presente, mas o gênero feminino "
                    "não aparece claramente na descrição."
                )

        if "running" in normalized_labels:
            running_terms = [
                "corre",
                "correndo",
            ]

            if not self._contains_any_word(
                normalized_description,
                running_terms,
            ):
                warnings.append(
                    "Informação possivelmente omitida: "
                    "a label 'running' está presente, mas a ação de correr "
                    "não aparece claramente na descrição."
                )

        if "walking" in normalized_labels:
            walking_terms = [
                "caminha",
                "caminhando",
                "anda",
                "andando",
            ]

            if not self._contains_any_word(
                normalized_description,
                walking_terms,
            ):
                warnings.append(
                    "Informação possivelmente omitida: "
                    "a label 'walking' está presente, mas a ação de caminhar "
                    "não aparece claramente na descrição."
                )

        if "glasses" in normalized_labels:
            if not self._contains_word(
                normalized_description,
                "oculos",
            ):
                warnings.append(
                    "Informação possivelmente omitida: "
                    "a label 'glasses' está presente, "
                    "mas os óculos não aparecem na descrição."
                )

        if "black_clothes" in normalized_labels:
            clothes_terms = [
                "roupas pretas",
                "roupa preta",
                "vestes pretas",
                "vestimenta preta",
            ]

            if not self._contains_any_phrase(
                normalized_description,
                clothes_terms,
            ):
                warnings.append(
                    "Informação possivelmente omitida: "
                    "a label 'black_clothes' está presente, "
                    "mas as roupas pretas não aparecem na descrição."
                )

        if "short_hair" in normalized_labels:
            hair_terms = [
                "cabelo curto",
                "cabelos curtos",
            ]

            if not self._contains_any_phrase(
                normalized_description,
                hair_terms,
            ):
                warnings.append(
                    "Informação possivelmente omitida: "
                    "a label 'short_hair' está presente, "
                    "mas o cabelo curto não aparece na descrição."
                )

        # ---------------------------------------------------------
        # Retorno
        # ---------------------------------------------------------

        return warnings

    def _extract_context_values(
        self,
        context: Dict,
    ) -> List[str]:
        values = []

        if not isinstance(context, dict):
            return values

        for value in context.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        values.append(item)

            elif isinstance(value, dict):
                values.extend(
                    self._extract_context_values(value)
                )

            elif isinstance(value, str):
                values.append(value)

        return values

    def _contains_word(
        self,
        text: str,
        word: str,
    ) -> bool:
        normalized_word = self._normalize(word)

        pattern = r"\b{}\b".format(
            re.escape(normalized_word)
        )

        return re.search(pattern, text) is not None

    def _contains_phrase(
        self,
        text: str,
        phrase: str,
    ) -> bool:
        normalized_phrase = self._normalize(phrase)

        return normalized_phrase in text

    def _contains_any_word(
        self,
        text: str,
        words: List[str],
    ) -> bool:
        for word in words:
            if self._contains_word(text, word):
                return True

        return False

    def _contains_any_phrase(
        self,
        text: str,
        phrases: List[str],
    ) -> bool:
        for phrase in phrases:
            if self._contains_phrase(text, phrase):
                return True

        return False

    def _normalize(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.strip().lower()

        text = unicodedata.normalize(
            "NFD",
            text,
        )

        text = "".join(
            char
            for char in text
            if unicodedata.category(char) != "Mn"
        )

        text = re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()