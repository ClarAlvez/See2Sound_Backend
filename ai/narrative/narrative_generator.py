from typing import Any, Dict, List, Optional

from ai.narrative.data_models import (
    NarrativeInput,
    NarrativeOutput,
    SceneContext,
    SpectraScene,
)
from ai.narrative.fidelity_filter import FidelityFilter
from ai.narrative.llm_client import LlamaCppClient
from ai.narrative.prompt_builder import NarrativePromptBuilder
from ai.narrative.redundancy_filter import RedundancyFilter
from ai.narrative.scene_context_builder import SceneContextBuilder


class LLMNarrativeGenerator:
    """
    Gerador narrativo da Sprint 5.

    Fluxo:
    Spectra labels
    ↓
    organização semântica da cena
    ↓
    prompt controlado
    ↓
    modelo local GGUF via llama.cpp
    ↓
    limpeza da resposta
    ↓
    filtro de fidelidade
    ↓
    filtro de redundância
    ↓
    timeline textual pronta para virar áudio
    """

    def __init__(
        self,
        model_path: str = "data/models/llama/Llama-3.2-1B-Instruct-Q6_K_L.gguf",
        similarity_threshold: float = 0.75,
        n_ctx: int = 5480,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
    ):
        self.client = LlamaCppClient(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            temperature=0.0,
            top_p=1.0,
            max_tokens=48,
            verbose=False,
        )

        self.scene_context_builder = SceneContextBuilder()
        self.prompt_builder = NarrativePromptBuilder()
        self.fidelity_filter = FidelityFilter()

        self.redundancy_filter = RedundancyFilter(
            label_similarity_threshold=similarity_threshold,
            text_similarity_threshold=0.85,
        )

    def generate(self, narrative_input: NarrativeInput) -> NarrativeOutput:
        cleaned_labels = self._clean_labels(narrative_input.labels)

        if not cleaned_labels:
            return NarrativeOutput(
                description="",
                start_time=narrative_input.start_time,
                end_time=narrative_input.end_time,
                labels=[],
                scene_context={},
                skipped=True,
                skip_reason="Nenhuma label recebida da Spectra.",
            )

        scene_context = self.scene_context_builder.build(cleaned_labels)
        scene_context_dict = self.scene_context_builder.to_prompt_dict(scene_context)

        prompt_data = self._build_prompt_data(
            narrative_input=narrative_input,
            cleaned_labels=cleaned_labels,
            scene_context_dict=scene_context_dict,
        )

        prompt = self.prompt_builder.build(prompt_data)

        raw_description = self.client.generate(prompt)
        description = self._clean_model_output(raw_description)

        if not description:
            fallback_description = (
                self._build_factual_fallback(
                    scene_context_dict
                )
            )

            if fallback_description:
                return NarrativeOutput(
                    description=(
                        fallback_description
                    ),
                    start_time=(
                        narrative_input.start_time
                    ),
                    end_time=(
                        narrative_input.end_time
                    ),
                    labels=cleaned_labels,
                    scene_context=(
                        scene_context_dict
                    ),
                    skipped=False,
                    skip_reason=None,
                )

            return NarrativeOutput(
                description="",
                start_time=(
                    narrative_input.start_time
                ),
                end_time=(
                    narrative_input.end_time
                ),
                labels=cleaned_labels,
                scene_context=(
                    scene_context_dict
                ),
                skipped=True,
                skip_reason=(
                    "O modelo não gerou "
                    "uma descrição válida."
                ),
            )

        fidelity_warnings = self.fidelity_filter.validate(
            description=description,
            labels=cleaned_labels,
            context=scene_context_dict,
        )

        critical_warnings = [
            warning
            for warning in fidelity_warnings
            if (
                warning.startswith("Possível detalhe inventado")
                or warning.startswith("Contradição")
                or warning.startswith("Ação inventada")
                or warning.startswith("Objeto inventado")
                or warning.startswith("Sujeito inventado")
            )
        ]

        if critical_warnings:
            fallback_description = (
                self._build_factual_fallback(
                    scene_context_dict
                )
            )

            if fallback_description:
                return NarrativeOutput(
                    description=(
                        fallback_description
                    ),
                    start_time=(
                        narrative_input.start_time
                    ),
                    end_time=(
                        narrative_input.end_time
                    ),
                    labels=cleaned_labels,
                    scene_context=(
                        scene_context_dict
                    ),
                    skipped=False,
                    skip_reason=None,
                    fidelity_warnings=(
                        fidelity_warnings
                    ),
                )

            return NarrativeOutput(
                description="",
                start_time=(
                    narrative_input.start_time
                ),
                end_time=(
                    narrative_input.end_time
                ),
                labels=cleaned_labels,
                scene_context=(
                    scene_context_dict
                ),
                skipped=True,
                skip_reason=(
                    "Descrição rejeitada "
                    "por inconsistência factual "
                    "e não foi possível gerar fallback."
                ),
                fidelity_warnings=(
                    fidelity_warnings
                ),
            )

        return NarrativeOutput(
            description=description,
            start_time=narrative_input.start_time,
            end_time=narrative_input.end_time,
            labels=cleaned_labels,
            scene_context=scene_context_dict,
            skipped=False,
            skip_reason=None,
            fidelity_warnings=fidelity_warnings,
        )
        

    def generate_batch(
        self,
        inputs: List[NarrativeInput],
        skip_similar_labels: bool = True,
        skip_similar_text: bool = True,
    ) -> List[NarrativeOutput]:
        outputs = []

        previous_labels = []
        previous_description = ""

        for item in inputs:
            current_labels = self._clean_labels(item.labels)

            if skip_similar_labels:
                labels_are_similar = self.redundancy_filter.is_too_similar_by_labels(
                    previous_labels=previous_labels,
                    current_labels=current_labels,
                )

                if labels_are_similar:
                    outputs.append(
                        NarrativeOutput(
                            description="",
                            start_time=item.start_time,
                            end_time=item.end_time,
                            labels=current_labels,
                            scene_context={},
                            skipped=True,
                            skip_reason="Cena muito parecida com a anterior.",
                        )
                    )
                    continue

            item.previous_description = previous_description

            output = self.generate(item)

            if output.skipped:
                outputs.append(output)
                continue

            if previous_description:
                same_text = self.redundancy_filter.is_exact_same_text(
                    previous_description=previous_description,
                    current_description=output.description,
                )

                similar_text = self.redundancy_filter.is_too_similar_by_text(
                    previous_description=previous_description,
                    current_description=output.description,
                )

                if same_text or (skip_similar_text and similar_text):
                    output.skipped = True
                    output.skip_reason = "Descrição muito parecida com a anterior."
                    output.description = ""

            outputs.append(output)

            if not output.skipped and output.description:
                previous_labels = current_labels
                previous_description = output.description

        return outputs

    def generate_from_spectra_scenes(
        self,
        spectra_scenes: List[SpectraScene],
    ) -> List[NarrativeOutput]:
        inputs = []

        for scene in spectra_scenes:
            inputs.append(
                NarrativeInput(
                    labels=scene.labels,
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    confidence=scene.confidence,
                    context=scene.context,
                )
            )

        return self.generate_batch(inputs)

    def generate_timeline_from_dicts(
        self,
        spectra_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        inputs = []

        for item in spectra_outputs:
            inputs.append(
                NarrativeInput(
                    labels=item.get(
                        "labels",
                        [],
                    ),
                    start_time=item.get(
                        "start_time"
                    ),
                    end_time=item.get(
                        "end_time"
                    ),
                    confidence=item.get(
                        "confidence",
                        {},
                    ),
                    context=item.get(
                        "context",
                        {},
                    ),
                )
            )

        print(
            "[Narrative DEBUG] "
            f"Entradas recebidas: {len(inputs)}"
        )

        outputs = self.generate_batch(
            inputs
        )

        print(
            "[Narrative DEBUG] "
            f"Saídas produzidas: {len(outputs)}"
        )

        valid_outputs = []

        for index, output in enumerate(
            outputs
        ):
            print(
                "\n"
                "[Narrative DEBUG] "
                f"Output #{index}"
            )

            print(
                "  labels:",
                output.labels,
            )

            print(
                "  skipped:",
                output.skipped,
            )

            print(
                "  motivo:",
                output.skip_reason,
            )

            print(
                "  descrição:",
                output.description,
            )

            print(
                "  fidelity_warnings:",
                output.fidelity_warnings,
            )

            if not output.skipped:
                valid_outputs.append(
                    output.to_dict()
                )

        print(
            "\n"
            "[Narrative DEBUG] "
            f"Descrições válidas: "
            f"{len(valid_outputs)}"
        )

        return valid_outputs

    def _build_prompt_data(
        self,
        narrative_input: NarrativeInput,
        cleaned_labels: List[str],
        scene_context_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "labels": cleaned_labels,
            "subjects": scene_context_dict.get("subjects", []),
            "actions": scene_context_dict.get("actions", []),
            "objects": scene_context_dict.get("objects", []),
            "environment": scene_context_dict.get("environment", []),
            "time": scene_context_dict.get("time", []),
            "attributes": scene_context_dict.get("attributes", []),
        }

    def _clean_labels(self, labels: List[str]) -> List[str]:
        cleaned_labels = []

        for label in labels:
            if not isinstance(label, str):
                continue

            clean_label = label.strip().lower()

            if clean_label and clean_label not in cleaned_labels:
                cleaned_labels.append(clean_label)

        return cleaned_labels

    def _clean_model_output(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        unwanted_prefixes = [
            "Audiodescrição:",
            "Descrição:",
            "Saída:",
            "Resposta:",
            "Frase:",
            "Texto:",
        ]

        for prefix in unwanted_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if lines:
            text = lines[0]

        text = text.strip()
        text = text.strip('"')
        text = text.strip("'")
        text = text.strip("“")
        text = text.strip("”")

        text = self._remove_list_marker(text)

        if not text:
            return ""

        if not text.endswith((".", "!", "?")):
            text += "."

        text = text[0].upper() + text[1:]

        return text

    def _remove_list_marker(self, text: str) -> str:
        markers = [
            "- ",
            "* ",
            "• ",
            "1. ",
            "2. ",
            "3. ",
        ]

        for marker in markers:
            if text.startswith(marker):
                return text[len(marker):].strip()

        return text
    
    def _build_factual_fallback(
        self,
        scene_context_dict: Dict[str, Any],
    ) -> str:
        subjects = scene_context_dict.get(
            "subjects",
            [],
        )

        actions = scene_context_dict.get(
            "actions",
            [],
        )

        objects = scene_context_dict.get(
            "objects",
            [],
        )

        environment = scene_context_dict.get(
            "environment",
            [],
        )

        time_labels = scene_context_dict.get(
            "time",
            [],
        )

        attributes = scene_context_dict.get(
            "attributes",
            [],
        )

        translations = (
            self.prompt_builder.LABEL_TO_PT
        )

        def translate(
            values: List[str],
        ) -> List[str]:
            return [
                translations.get(
                    value,
                    value.replace(
                        "_",
                        " ",
                    ),
                )
                for value in values
            ]

        subjects_pt = translate(
            subjects
        )

        actions_pt = translate(
            actions
        )

        objects_pt = translate(
            objects
        )

        environment_pt = translate(
            environment
        )

        time_pt = translate(
            time_labels
        )

        attributes_pt = translate(
            attributes
        )

        parts = []

        # --------------------------------------------------------
        # Sujeito
        # --------------------------------------------------------

        if subjects_pt:
            parts.append(
                subjects_pt[0]
            )

            if actions_pt:
                parts.append(
                    actions_pt[0]
                )

            if objects_pt:
                parts.append(
                    "com "
                    + objects_pt[0]
                )

        # --------------------------------------------------------
        # Cena sem sujeito
        # --------------------------------------------------------

        else:
            if environment:
                parts.append(
                    self._describe_environment(
                        environment
                    )
                )

            elif objects_pt:
                parts.append(
                    "Há "
                    + ", ".join(
                        objects_pt
                    )
                )

            elif attributes_pt:
                parts.append(
                    "O ambiente está "
                    + ", ".join(
                        attributes_pt
                    )
                )

        # --------------------------------------------------------
        # Tempo / atmosfera
        # --------------------------------------------------------

        if time_pt:
            parts.append(
                time_pt[0]
            )

        if (
            attributes_pt
            and subjects_pt
        ):
            parts.append(
                attributes_pt[0]
            )

        text = " ".join(
            part.strip()
            for part in parts
            if part
        ).strip()

        if not text:
            return ""

        text = (
            text[0].upper()
            + text[1:]
        )

        if not text.endswith(
            (".", "!", "?")
        ):
            text += "."

        return text
    
    def _describe_environment(
        self,
        environment: List[str],
    ) -> str:
        environment_set = set(
            environment
        )

        if (
            "outdoor" in environment_set
            and "ocean" in environment_set
        ):
            return (
                "ambiente ao ar livre "
                "com oceano"
            )

        if (
            "outdoor" in environment_set
            and "park" in environment_set
        ):
            return (
                "ambiente ao ar livre "
                "em um parque"
            )

        if (
            "street" in environment_set
            and "city" in environment_set
        ):
            return (
                "ambiente ao ar livre "
                "em uma rua da cidade"
            )

        if "ocean" in environment_set:
            return (
                "o oceano compõe o cenário"
            )

        if "park" in environment_set:
            return (
                "o cenário é um parque"
            )

        if "street" in environment_set:
            return (
                "o cenário é uma rua"
            )

        if "city" in environment_set:
            return (
                "o cenário é uma cidade"
            )

        if "outdoor" in environment_set:
            return (
                "o cenário é ao ar livre"
            )

        if "indoor" in environment_set:
            return (
                "o cenário é um ambiente interno"
            )

        translated = (
            self.prompt_builder
            .translate_values(
                environment
            )
        )

        if translated:
            return (
                "o cenário apresenta "
                + ", ".join(
                    translated
                )
            )

        return ""
