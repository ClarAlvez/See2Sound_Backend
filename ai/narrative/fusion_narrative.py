
from ai.narrative.narrative_generator import (
    LLMNarrativeGenerator,
)


class FusionNarrativeGenerator(
    LLMNarrativeGenerator
):
    """
    Extensão conservadora da Narrative.

    Preserva:
      - Qwen
      - Continuidade
      - Memória
      - Reescrita segura
      - Tratamento de pausas

    Adiciona:
      - Prioridade da fusão
      - Foco no veículo quando apropriado
      - Prevenção de associação entre
        ação de veículo e animal
    """

    VEHICLE_PRIORITY = (
        "car",
        "bus",
        "truck",
        "motorcycle",
    )

    def _select_facts(
        self,
        narrative_input,
    ):
        facts = super()._select_facts(
            narrative_input
        )

        context = (
            narrative_input.context
            or {}
        )

        labels = set(
            narrative_input.labels
            or []
        )

        preferred = context.get(
            "preferred_action"
        )

        # Verificamos se a ação preferida
        # está realmente nas labels finais.
        if (
            preferred
            and preferred in self.ACTIONS
            and preferred in labels
        ):
            facts["action"] = preferred

            print(
                "[Narrative Fusion] "
                "Ação priorizada:",
                preferred,
            )

        # No caso de veículos, podemos
        # descrever seu deslocamento sem
        # afirmar que o motorista é visível.
        vehicle_hint = context.get(
            "vehicle_action_hint", {}
        )

        is_driving = (
            facts.get("action")
            == "driving"
            and vehicle_hint.get(
                "action"
            ) == "driving"
        )

        confidence = (
            narrative_input.confidence
            or {}
        )

        if is_driving:
            vehicle = None
            vehicle_score = 0.0

            for candidate in (
                self.VEHICLE_PRIORITY
            ):
                if candidate not in labels:
                    continue

                score = float(
                    confidence.get(
                        f"object.{candidate}",
                        0.0,
                    )
                )

                if score > vehicle_score:
                    vehicle = candidate
                    vehicle_score = score

            # Somente usar foco no veículo
            # quando a detecção é forte.
            if (
                vehicle is not None
                and vehicle_score >= 0.70
            ):
                person_labels = (
                    self.HUMAN_SUBJECTS
                )

                has_person = bool(
                    labels & person_labels
                )

                if not has_person:
                    facts["subject"] = None
                    facts["vehicle_focus"] = (
                        vehicle
                    )
                    facts["vehicle_score"] = (
                        vehicle_score
                    )

                    # Não afirmar que o animal
                    # está dirigindo ou parado.
                    # O animal continua nos dados
                    # originais, para futura
                    # verificação de Objects.
                    facts["action"] = "driving"

                    facts["objects"] = [
                        item
                        for item in facts.get(
                            "objects", []
                        )
                        if item != vehicle
                        and item != "animal"
                    ]

                    print(
                        "[Narrative Fusion] "
                        "Foco no veículo:",
                        vehicle,
                    )

        return facts

    def _build_base_description(
        self,
        facts,
    ):
        vehicle = facts.get(
            "vehicle_focus"
        )

        action = facts.get(
            "action"
        )

        if (
            not vehicle
            or action != "driving"
        ):
            return (
                super()
                ._build_base_description(
                    facts
                )
            )

        vehicle_names = {
            "car": "Um carro",
            "truck": "Um caminhão",
            "bus": "Um ônibus",
            "motorcycle": "Uma motocicleta",
        }

        subject = vehicle_names.get(
            vehicle,
            "Um veículo",
        )

        # Não atribuir condutor humano
        # sem evidência visual direta.
        text = (
            f"{subject} aparece em movimento"
        )

        environment = facts.get(
            "environment"
        )

        if environment == "road":
            text += " em uma estrada"

        elif environment == "street":
            text += " em uma rua"

        elif (
            environment
            and environment
            in self.ENVIRONMENTS
        ):
            text += (
                " "
                + self.ENVIRONMENTS[
                    environment
                ]
            )

        text = (
            self._append_scene_modifiers(
                text,
                facts,
            )
        )

        return (
            text.rstrip(".")
            + "."
        )
