"""Adaptador que preserva a FusionNarrativeGenerator já existente."""
from dataclasses import replace

from ai.narrative.fusion_narrative import FusionNarrativeGenerator


ATTR_TEXT = {
    "glasses": "usa óculos", "hat": "usa chapéu", "backpack": "usa mochila",
    "handbag": "carrega uma bolsa", "long_hair": "tem cabelos longos",
    "short_hair": "tem cabelos curtos", "black_clothes": "veste roupas pretas",
    "white_clothes": "veste roupas brancas", "red_clothes": "veste roupas vermelhas",
    "blue_clothes": "veste roupas azuis", "green_clothes": "veste roupas verdes",
}

PLURAL_ACTIONS = {
    "walking": "caminham", "running": "correm", "cycling": "andam de bicicleta",
    "cooking": "cozinham", "standing": "estão em pé", "sitting": "estão sentadas",
    "working": "trabalham", "talking": "conversam", "dancing": "dançam",
    "driving": "dirigem", "moving": "se movimentam",
}

# Existência de uma ação entre as pessoas, sem universalizar nem identificar
# quem a executa. "working" não comprova ocupação ou trabalho profissional.
GROUP_ACTIONS = {
    "walking": "há quem caminhe", "running": "há quem corra",
    "cycling": "há quem ande de bicicleta", "cooking": "há quem cozinhe",
    "standing": "há quem esteja em pé", "sitting": "há quem esteja sentado",
    "still": "há quem permaneça parado",
    "working": "há quem realize uma atividade", "talking": "há quem converse",
    "dancing": "há quem dance", "moving": "há quem se movimente",
    "computer_use": "há quem use computador", "writing": "há quem escreva",
    "reading": "há quem leia", "phone_use": "há quem use telefone",
    "driving": "há quem dirija",
    "eating": "há quem coma", "drinking": "há quem beba",
    "exercising": "há quem se exercite", "cleaning": "há quem faça uma limpeza",
    "instrument_playing": "há quem toque um instrumento",
}


class PersonFusionNarrativeGenerator(FusionNarrativeGenerator):
    def _select_facts(self, narrative_input):
        person_data = (narrative_input.context or {}).get("person_intermediate")
        is_group = (isinstance(person_data, dict)
                    and person_data.get("status") == "analyzed"
                    and person_data.get("count", 0) >= 2)
        if is_group:
            # Defesa também para chamadas diretas que não passaram pela cópia
            # scene_for_narrative. Não alterar labels/confidências do chamador.
            hidden = set((narrative_input.context or {}).get("suppressed_object_labels", []))
            narrative_input = replace(
                narrative_input,
                labels=[label for label in narrative_input.labels if label not in hidden],
                confidence={key: value for key, value in narrative_input.confidence.items()
                            if key not in {f"object.{label}" for label in hidden}},
            )
        facts = super()._select_facts(narrative_input)
        if facts.get("vehicle_focus"):
            return facts
        if not isinstance(person_data, dict) or person_data.get("status") != "analyzed":
            return facts
        facts["person_group"] = person_data
        if person_data.get("count", 0) >= 2:
            # Atributos globais vêm da união de pessoas diferentes.
            facts["appearance"] = []
            # Score global/união de crops não confirma ação coletiva. Uma
            # preferência auditada pode superar a postura de maior score.
            context = narrative_input.context or {}
            preferred = context.get("preferred_action")
            evidence = context.get("preferred_action_evidence") or {}
            confirmed = (
                preferred in GROUP_ACTIONS
                and context.get("preferred_action_source") == "xclip_temporal_consensus"
                and evidence.get("label") == preferred
                and evidence.get("evidence_passed") is True
                and evidence.get("person_supported_clips", 0) > 0
                and self._accepted(preferred, set(narrative_input.labels),
                                   narrative_input.confidence, threshold=0.60)
            )
            facts["action"] = preferred if confirmed else None
            facts["subject"] = "person"
            crops = (context.get("raw_outputs", {}).get("object", {})
                     .get("crop_results", []))
            laptop = any(crop.get("detector_label") == "laptop"
                         and {"computer", "screen"}.issubset(crop.get("mapped_labels", []))
                         for crop in crops)
            separate_screen = any(crop.get("detector_label") != "laptop"
                                  and "screen" in crop.get("mapped_labels", [])
                                  for crop in crops)
            facts["same_computer_screen"] = laptop and not separate_screen
            # Uma classificação Scene isolada não confirma a função escolar do
            # local. Usar somente o ambiente genérico já reconhecido, sem office.
            if facts.get("environment") == "school":
                facts["environment"] = (
                    "indoor" if self._accepted("indoor", set(narrative_input.labels),
                                               narrative_input.confidence, threshold=0.45)
                    else None
                )
        elif person_data.get("count") == 1:
            facts["appearance"] = []
            facts["subject"] = "person"
        return facts

    def _group_objects(self, facts):
        objects = list(dict.fromkeys(facts.get("objects", [])))
        # A seleção continua intacta; omitir só a segunda menção ao mesmo laptop.
        if facts.get("same_computer_screen") and "computer" in objects:
            objects = [label for label in objects if label != "screen"]
        return [self.OBJECTS[label] for label in objects if label in self.OBJECTS][:2]

    def _people_text(self, facts):
        group = facts.get("person_group")
        if not group:
            return ""
        count, kind = group.get("count", 0), group.get("count_type")
        if kind == "exact" and count == 1:
            start = "Uma pessoa"
        elif kind == "exact" and count == 2:
            start = "Duas pessoas"
        elif kind == "group":
            start = "Um grupo de pessoas"
        else:
            start = "Algumas pessoas"
        details = []
        for i, person in enumerate(group.get("people", [])[:3]):
            descriptions = [ATTR_TEXT[item["label"]] for item in person.get("attributes", [])
                            if item.get("label") in ATTR_TEXT]
            if descriptions:
                name = ("A pessoa" if count == 1 else f"A pessoa {i+1}")
                details.append(name + " " + " e ".join(descriptions[:2]))
        action = facts.get("action")
        if count == 1 and action in self.ACTIONS:
            start += " " + self._action_text("person", action)
        environment = facts.get("environment")
        if environment in self.ENVIRONMENTS:
            start += " " + self.ENVIRONMENTS[environment]
        if count >= 2:
            objects = self._group_objects(facts)
            if objects:
                start += ", com " + " e ".join(objects) + " na cena"
            if action in GROUP_ACTIONS:
                start += "; entre elas, " + GROUP_ACTIONS[action]
        start = self._append_scene_modifiers(start, facts).rstrip(".") + "."
        return start + (" " + ". ".join(details) + "." if details else "")

    def _build_base_description(self, facts):
        if facts.get("person_group"):
            return self._people_text(facts)
        return super()._build_base_description(facts)

    def _build_delta_description(self, current, previous):
        # A base de continuidade original não considera contagens.
        if current.get("person_group"):
            now = current["person_group"]
            old = (previous or {}).get("person_group") or {}
            if previous is None or (now.get("count_type"), now.get("count")) != (old.get("count_type"), old.get("count")):
                return self._build_base_description(current)
            # Não perder novos objetos/atividades com a mesma contagem. A
            # assinatura contém somente fatos verbalizados, sem bboxes/scores.
            if now.get("count", 0) >= 2 and self._people_text(current) != self._people_text(previous):
                return self._build_base_description(current)
            return ""
        return super()._build_delta_description(current, previous)
