
import re
import unicodedata

from typing import Any, Dict, List, Optional

from ai.narrative.data_models import (
    NarrativeInput,
    NarrativeOutput,
    SpectraScene,
)
from ai.narrative.llm_client import LlamaCppClient


class LLMNarrativeGenerator:

    SUBJECTS = {
        "woman": "Uma mulher",
        "man": "Um homem",
        "girl": "Uma menina",
        "boy": "Um menino",
        "child": "Uma criança",
        "dog": "Um cachorro",
        "cat": "Um gato",
        "person": "Uma pessoa",
    }

    SUBJECT_PRIORITY = [
        "woman", "man", "girl", "boy",
        "child", "dog", "cat", "person"
    ]

    ACTIONS = {
        "walking": "caminha",
        "running": "corre",
        "standing": "está em pé",
        "sitting": "está sentada",
    }

    APPEARANCE = {
        "long_hair": "de cabelos longos",
        "short_hair": "de cabelos curtos",
        "black_clothes": "de roupas pretas",
        "white_clothes": "de roupas brancas",
        "glasses": "de óculos",
    }

    ENVIRONMENTS = {
        "street": "em uma rua",
        "road": "em uma estrada",
        "city": "em uma cidade",
        "outdoor": "ao ar livre",
        "indoor": "em um ambiente interno",
    }

    OBJECTS = {
        "umbrella": "um guarda-chuva",
        "backpack": "uma mochila",
        "bicycle": "uma bicicleta",
        "car": "um carro",
        "window": "uma janela",
    }

    def __init__(
        self,
        model_path: str = (
            "data/models/qwen/"
            "qwen2.5-3b-instruct-q4_k_m.gguf"
        ),
        similarity_threshold: float = 0.75,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        use_llm: bool = False,
    ):
        self.use_llm = use_llm
        self.client = None

        if self.use_llm:
            self.client = LlamaCppClient(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                temperature=0.1,
                top_p=0.8,
                max_tokens=80,
                verbose=False,
            )

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize(
            "NFD",
            text.lower()
        )

        text = "".join(
            character
            for character in text
            if unicodedata.category(character) != "Mn"
        )

        return re.sub(
            r"\s+", " ",
            re.sub(r"[^\w\s]", " ", text)
        ).strip()

    def _get_confidence(
        self,
        label: str,
        confidence: Dict[str, Any],
    ) -> Optional[float]:

        if not isinstance(confidence, dict):
            return None

        scores = []

        for key, value in confidence.items():
            if (
                key == label
                or key.endswith("." + label)
            ):
                try:
                    scores.append(float(value))
                except (TypeError, ValueError):
                    pass

        return max(scores) if scores else None

    def _select_facts(
        self,
        narrative_input: NarrativeInput,
    ) -> Dict[str, Any]:

        labels = set(
            label.strip().lower()
            for label in narrative_input.labels
            if isinstance(label, str)
        )

        confidence = narrative_input.confidence or {}

        def accepted(label, threshold):
            if label not in labels:
                return False

            score = self._get_confidence(
                label,
                confidence
            )

            if score is None:
                return True

            return score >= threshold

        subject = None

        for label in self.SUBJECT_PRIORITY:
            if accepted(label, 0.65):
                subject = label
                break

        action = None

        if subject is not None:
            action_candidates = [
                label
                for label in self.ACTIONS
                if accepted(label, 0.70)
            ]

            if action_candidates:
                action = max(
                    action_candidates,
                    key=lambda label: (
                        self._get_confidence(
                            label,
                            confidence
                        ) or 0.0
                    )
                )

        appearance = []

        if subject in {
            "person", "woman", "man",
            "boy", "girl", "child"
        }:
            for label in [
                "long_hair",
                "short_hair",
                "black_clothes",
                "white_clothes",
                "glasses",
            ]:
                if accepted(label, 0.72):
                    appearance.append(label)

            if (
                "long_hair" in appearance
                and "short_hair" in appearance
            ):
                appearance = [
                    label
                    for label in appearance
                    if label != "short_hair"
                ]

            if (
                "black_clothes" in appearance
                and "white_clothes" in appearance
            ):
                best_color = max(
                    ["black_clothes", "white_clothes"],
                    key=lambda label: (
                        self._get_confidence(
                            label, confidence
                        ) or 0
                    )
                )

                appearance = [
                    label
                    for label in appearance
                    if label not in {
                        "black_clothes",
                        "white_clothes"
                    } or label == best_color
                ]

        environment = None

        for label in [
            "street", "road", "city",
            "indoor", "outdoor"
        ]:
            if accepted(label, 0.70):
                environment = label
                break

        objects = []

        for label in [
            "umbrella",
            "backpack",
            "bicycle",
            "car",
            "window",
        ]:
            if accepted(label, 0.80):
                objects.append(label)

        atmosphere = None

        if accepted("rainy", 0.80):
            atmosphere = "rainy"
        elif accepted("cloudy", 0.80):
            atmosphere = "cloudy"

        return {
            "subject": subject,
            "action": action,
            "appearance": appearance[:2],
            "environment": environment,
            "objects": objects[:2],
            "atmosphere": atmosphere,
        }

    def _build_base_description(
        self,
        facts: Dict[str, Any],
    ) -> str:

        subject = facts["subject"]
        action = facts["action"]
        appearance = facts["appearance"]
        environment = facts["environment"]
        objects = facts["objects"]
        atmosphere = facts["atmosphere"]

        if subject:
            text = self.SUBJECTS[subject]

            if appearance:
                text += " " + " e ".join(
                    self.APPEARANCE[label]
                    for label in appearance
                )

            if action:
                action_text = self.ACTIONS[action]

                if action == "sitting":
                    if subject in {"man", "boy"}:
                        action_text = "está sentado"
                    elif subject in {"dog", "cat"}:
                        action_text = "está sentado"

                text += " " + action_text

            else:
                text += " aparece"

            if environment:
                text += " " + self.ENVIRONMENTS[
                    environment
                ]

            if objects:
                text += ", enquanto " + (
                    self.OBJECTS[objects[0]]
                    + " também aparece na cena"
                )

            return text + "."

        if environment:
            if environment == "street":
                text = "Uma rua aparece na cena"
            elif environment == "road":
                text = "Uma estrada aparece na cena"
            elif environment == "city":
                text = "Uma cidade aparece na cena"
            elif environment == "indoor":
                text = "A cena mostra um ambiente interno"
            else:
                text = "A cena mostra um ambiente externo"

            if atmosphere == "rainy":
                text += ", em meio à chuva"
            elif atmosphere == "cloudy":
                text += ", sob tempo nublado"

            return text + "."

        if objects:
            object_text = self.OBJECTS[objects[0]]

            return (
                object_text[0].upper()
                + object_text[1:]
                + " aparece na cena."
            )

        return ""

    def _safe_llm_rewrite(
        self,
        base_description: str,
    ) -> str:

        if not self.use_llm or self.client is None:
            return base_description

        prompt = (
            "Reescreva esta frase em português natural. "
            "Não acrescente nem retire fatos. "
            "Não mude sujeitos, ações ou objetos. "
            "Retorne somente uma frase.\n"
            f"Frase: {base_description}"
        )

        try:
            raw = self.client.generate(prompt)
            candidate = raw.strip()

            if not candidate:
                return base_description

            if any(
                char in candidate
                for char in "{}[]\n"
            ):
                return base_description

            if len(re.findall(
                r"[.!?]", candidate
            )) > 1:
                return base_description

            base_words = self._normalize(
                base_description
            ).split()

            candidate_words = self._normalize(
                candidate
            ).split()

            allowed_connectors = {
                "a", "o", "as", "os",
                "um", "uma", "de", "do",
                "da", "em", "no", "na",
                "e", "com", "ao"
            }

            allowed_words = (
                set(base_words)
                | allowed_connectors
            )

            if any(
                word not in allowed_words
                for word in candidate_words
            ):
                return base_description

            required_words = {
                word
                for word in base_words
                if word not in allowed_connectors
            }

            if not required_words.issubset(
                set(candidate_words)
            ):
                return base_description

            return candidate

        except Exception as error:
            print(
                "[Narrative LLM] "
                f"Reformulação ignorada: {error}"
            )

            return base_description

    def generate(
        self,
        narrative_input: NarrativeInput,
    ) -> NarrativeOutput:

        labels = list(dict.fromkeys(
            label.strip().lower()
            for label in narrative_input.labels
            if isinstance(label, str)
            and label.strip()
        ))

        facts = self._select_facts(
            narrative_input
        )

        base_description = (
            self._build_base_description(facts)
        )

        description = base_description

        if not description:
            return NarrativeOutput(
                description="",
                start_time=narrative_input.start_time,
                end_time=narrative_input.end_time,
                labels=labels,
                scene_context=facts,
                skipped=True,
                skip_reason=(
                    "Fatos insuficientes para "
                    "uma descrição confiável."
                ),
            )

        return NarrativeOutput(
            description=description,
            start_time=narrative_input.start_time,
            end_time=narrative_input.end_time,
            labels=labels,
            scene_context=facts,
            skipped=False,
            skip_reason=None,
            fidelity_warnings=[],
        )
        
    
    def _build_delta_description(
        self,
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
    ) -> str:

        subject = current.get("subject")
        action = current.get("action")
        appearance = current.get("appearance", [])
        environment = current.get("environment")
        objects = current.get("objects", [])
        atmosphere = current.get("atmosphere")

        if not subject:
            return ""

        if previous is None:
            return self._build_base_description(current)

        old_subject = previous.get("subject")
        old_action = previous.get("action")
        old_appearance = previous.get("appearance", [])
        old_environment = previous.get("environment")
        old_objects = previous.get("objects", [])
        old_atmosphere = previous.get("atmosphere")

        human_subjects = {
            "person", "woman", "man",
            "girl", "boy", "child"
        }

        same_category = (
            subject == old_subject
            or (
                subject in human_subjects
                and old_subject in human_subjects
            )
        )

        if not same_category:
            return self._build_base_description(current)

        new_appearance = [
            item
            for item in appearance
            if item not in old_appearance
        ]

        new_objects = [
            item
            for item in objects
            if item not in old_objects
        ]

        action_changed = (
            action is not None
            and action != old_action
        )

        environment_changed = (
            environment is not None
            and environment != old_environment
        )

        atmosphere_changed = (
            atmosphere is not None
            and atmosphere != old_atmosphere
        )

        subject_refined = (
            old_subject == "person"
            and subject in {
                "woman", "man", "girl",
                "boy", "child"
            }
        )

        if not any([
            action_changed,
            environment_changed,
            atmosphere_changed,
            new_appearance,
            new_objects,
            subject_refined,
        ]):
            return ""

        if subject_refined:
            references = {
                "woman": "A pessoa, identificada como mulher,",
                "man": "A pessoa, identificada como homem,",
                "girl": "A pessoa, identificada como menina,",
                "boy": "A pessoa, identificada como menino,",
                "child": "A pessoa, identificada como criança,",
            }

            text = references.get(
                subject,
                self.SUBJECTS[subject]
            )
        else:
            references = {
                "woman": "A mulher",
                "man": "O homem",
                "girl": "A menina",
                "boy": "O menino",
                "child": "A criança",
                "person": "A pessoa",
                "dog": "O cachorro",
                "cat": "O gato",
            }

            text = references.get(
                subject,
                self.SUBJECTS.get(
                    subject, "Uma pessoa"
                )
            )

        if new_appearance:
            details = [
                self.APPEARANCE[item]
                for item in new_appearance
                if item in self.APPEARANCE
            ]

            if details:
                text += " " + " e ".join(details)

        if action_changed:
            verb = self.ACTIONS[action]

            if action == "sitting" and subject in {
                "man", "boy", "dog", "cat"
            }:
                verb = "está sentado"

            text += " " + verb

        else:
            if environment_changed:
                text += " aparece"
            elif new_objects:
                text += " aparece novamente"
            else:
                text += " é vista"

        if environment_changed:
            text += " " + self.ENVIRONMENTS[environment]

        if new_objects:
            names = [
                self.OBJECTS[item]
                for item in new_objects
                if item in self.OBJECTS
            ]

            if names:
                if len(names) == 1:
                    text += (
                        ", enquanto "
                        + names[0]
                        + " surge na imagem"
                    )
                else:
                    text += (
                        ", com "
                        + " e ".join(names)
                        + " também visíveis"
                    )

        if atmosphere_changed:
            if atmosphere == "rainy":
                text += ", em meio à chuva"
            elif atmosphere == "cloudy":
                text += ", sob tempo nublado"

        return text.strip() + "."

    
    
    def generate_batch(
        self,
        inputs: List[NarrativeInput],
        skip_similar_labels: bool = True,
        skip_similar_text: bool = True,
    ) -> List[NarrativeOutput]:

        outputs = []

        last_narrated_facts = None
        last_narrated_end = None
        last_description = ""

        narrated_appearance = set()
        narrated_objects = set()

        max_memory_gap = 8.0
        time_tolerance = 0.15

        for index, item in enumerate(inputs):
            output = self.generate(item)
            current_facts = output.scene_context

            if output.skipped:
                outputs.append(output)
                continue

            if current_facts.get("subject") is None:
                output.skipped = True
                output.description = ""
                output.skip_reason = (
                    "Cena sem sujeito principal "
                    "ou mudança confirmada."
                )
                outputs.append(output)
                continue

            gap = None

            if (
                last_narrated_end is not None
                and item.start_time is not None
            ):
                gap = (
                    float(item.start_time)
                    - float(last_narrated_end)
                )

            memory_valid = (
                last_narrated_facts is not None
                and gap is not None
                and -time_tolerance <= gap <= max_memory_gap
            )

            previous = (
                last_narrated_facts
                if memory_valid
                else None
            )

            print(
                f"\n[Narrative MEMORY] Cena {index + 1}"
            )
            print("  Início:", item.start_time)
            print("  Último fim:", last_narrated_end)
            print("  Diferença:", gap)
            print("  Memória válida:", memory_valid)

            if previous is None:
                narrated_appearance.clear()
                narrated_objects.clear()

            comparison_facts = None

            if previous is not None:
                comparison_facts = dict(previous)
                comparison_facts["appearance"] = list(
                    narrated_appearance
                )
                comparison_facts["objects"] = list(
                    narrated_objects
                )

            description = self._build_delta_description(
                current_facts,
                comparison_facts
            )

            if not description:
                output.skipped = True
                output.description = ""
                output.skip_reason = (
                    "Nenhuma informação nova relevante."
                )
                outputs.append(output)
                continue

            if (
                last_description
                and self._normalize(description)
                == self._normalize(last_description)
            ):
                output.skipped = True
                output.description = ""
                output.skip_reason = (
                    "Descrição repetida."
                )
                outputs.append(output)
                continue

            if self.use_llm:
                description = self._safe_llm_rewrite(
                    description
                )

            output.description = description

            output.scene_context = {
                **current_facts,
                "narrative_mode": (
                    "delta" if previous else "initial"
                ),
                "reference_scene": (
                    last_narrated_end
                    if previous else None
                ),
            }

            last_narrated_facts = dict(
                current_facts
            )

            last_narrated_end = item.end_time
            last_description = description

            narrated_appearance.update(
                current_facts.get("appearance", [])
            )
            narrated_objects.update(
                current_facts.get("objects", [])
            )

            outputs.append(output)

            print(
                "  Modo:",
                output.scene_context["narrative_mode"]
            )
            print("  Descrição:", description)

        return outputs

    def generate_from_spectra_scenes(
        self,
        spectra_scenes: List[SpectraScene],
    ) -> List[NarrativeOutput]:

        return self.generate_batch([
            NarrativeInput(
                labels=scene.labels,
                start_time=scene.start_time,
                end_time=scene.end_time,
                confidence=scene.confidence,
                context=scene.context,
            )
            for scene in spectra_scenes
        ])

    def generate_timeline_from_dicts(
        self,
        spectra_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        inputs = [
            NarrativeInput(
                labels=item.get("labels", []),
                start_time=item.get("start_time"),
                end_time=item.get("end_time"),
                confidence=item.get("confidence", {}),
                context=item.get("context", {}),
            )
            for item in spectra_outputs
        ]

        print(
            "[Narrative DEBUG] "
            f"Entradas recebidas: {len(inputs)}"
        )

        outputs = self.generate_batch(inputs)

        valid_outputs = []

        for index, output in enumerate(outputs):
            print(
                f"\n[Narrative DEBUG] Cena {index + 1}"
            )
            print(
                "  fatos selecionados:",
                output.scene_context
            )
            print("  descrição:", output.description)
            print("  skipped:", output.skipped)
            print("  motivo:", output.skip_reason)

            if not output.skipped:
                valid_outputs.append(
                    output.to_dict()
                )

        print(
            "[Narrative DEBUG] "
            f"Descrições válidas: {len(valid_outputs)}"
        )

        return valid_outputs
