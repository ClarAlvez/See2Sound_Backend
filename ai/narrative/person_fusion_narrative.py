"""Adaptador que preserva a FusionNarrativeGenerator já existente."""
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


class PersonFusionNarrativeGenerator(FusionNarrativeGenerator):
    def _select_facts(self, narrative_input):
        facts = super()._select_facts(narrative_input)
        person_data = (narrative_input.context or {}).get("person_intermediate")
        if not isinstance(person_data, dict) or person_data.get("status") != "analyzed":
            return facts
        facts["person_group"] = person_data
        if person_data.get("count", 0) >= 2:
            # Atributos globais vêm da união de pessoas diferentes.
            facts["appearance"] = []
            # Não afirmar que todos executam uma ação vista globalmente.
            facts["action"] = None
            facts["subject"] = "person"
        elif person_data.get("count") == 1:
            facts["appearance"] = []
            facts["subject"] = "person"
        return facts

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
        elif count >= 2 and facts.get("collective_action_confirmed") and action in PLURAL_ACTIONS:
            start += " " + PLURAL_ACTIONS[action]
        environment = facts.get("environment")
        if environment in self.ENVIRONMENTS:
            start += " " + self.ENVIRONMENTS[environment]
        start = self._append_scene_modifiers(start, facts).rstrip(".") + "."
        return start + (" " + ". ".join(details) + "." if details else "")

    def _build_base_description(self, facts):
        if facts.get("person_group"):
            return self._people_text(facts)
        return super()._build_base_description(facts)

    def _build_delta_description(self, current, previous):
        # A base de continuidade original não considera contagens.
        # Recontar somente quando há transição de tipo/quantidade; para
        # o restante manter a memória e o filtro da Narrative anterior.
        if current.get("person_group"):
            now = current["person_group"]
            old = (previous or {}).get("person_group") or {}
            if previous is None or (now.get("count_type"), now.get("count")) != (old.get("count_type"), old.get("count")):
                return self._build_base_description(current)
            return ""
        return super()._build_delta_description(current, previous)
