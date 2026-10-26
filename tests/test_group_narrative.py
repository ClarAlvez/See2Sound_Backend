"""Narrative regressions using saved facts and stubs; no model/media execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def narrative(monkeypatch):
    # Load only the real narrative logic. Even importing llama_cpp is forbidden.
    for name, directory in (("ai", ROOT / "ai"), ("ai.narrative", ROOT / "ai/narrative")):
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, name, package)
    client_module = ModuleType("ai.narrative.llm_client")

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Tests must never construct a model client")

    client_module.LlamaCppClient = ForbiddenClient
    monkeypatch.setitem(sys.modules, client_module.__name__, client_module)
    loaded = {}
    for short_name in ("data_models", "narrative_generator", "fusion_narrative", "person_fusion_narrative"):
        name = "ai.narrative." + short_name
        spec = importlib.util.spec_from_file_location(name, ROOT / "ai/narrative" / (short_name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[short_name] = module
    return SimpleNamespace(
        generator=loaded["person_fusion_narrative"].PersonFusionNarrativeGenerator(use_llm=False),
        base=loaded["fusion_narrative"].FusionNarrativeGenerator(use_llm=False),
        Input=loaded["data_models"].NarrativeInput,
    )


def scene(narrative, objects=True):
    labels = ["person", "indoor", "still", "working"]
    confidence = {"person.person": 1., "scene.indoor": .99,
                  "action.still": .9623, "action.working": .9492}
    context = {"person_intermediate": {"status": "analyzed", "count": 3,
                                       "count_type": "group", "people": []}}
    if objects:
        labels += ["computer", "screen"]
        confidence.update({"object.computer": .5944, "object.screen": .5944})
        context["raw_outputs"] = {"object": {"crop_results": [
            {"detector_label": "laptop", "mapped_labels": ["computer", "screen"]}]}}
    return narrative.Input(labels=labels, confidence=confidence, context=context,
                           start_time=0., end_time=5.)


def confirm(item, label="writing"):
    if label not in item.labels:
        item.labels.append(label)
    item.confidence["action." + label] = .70
    item.context.update(preferred_action=label,
                        preferred_action_source="xclip_temporal_consensus",
                        preferred_action_evidence={"label": label, "evidence_passed": True,
                                                   "person_supported_clips": 4})


def test_group_includes_selected_computer_without_claiming_use(narrative):
    item = scene(narrative)
    original = copy.deepcopy(item)
    result = narrative.generator.generate(item)
    assert result.scene_context["objects"] == ["computer", "screen"]
    assert "um computador na cena" in result.description
    assert result.scene_context["action"] is None
    assert "use" not in result.description and "trabalh" not in result.description
    assert item == original


def test_group_without_relevant_objects(narrative):
    item = scene(narrative, objects=False)
    item.labels.append("cup")
    item.confidence["object.cup"] = .2
    result = narrative.generator.generate(item)
    assert result.scene_context["objects"] == []
    assert "computador" not in result.description and "xícara" not in result.description


def test_suppression_before_selection_cannot_be_undone(narrative):
    item = scene(narrative)
    item.context["suppressed_object_labels"] = ["computer", "screen"]
    result = narrative.generator.generate(item)
    assert result.scene_context["objects"] == []
    assert "computador" not in result.description and "tela" not in result.description
    assert "computer" in item.labels


def test_same_laptop_is_not_verbalized_twice(narrative):
    result = narrative.generator.generate(scene(narrative))
    assert result.description.count("computador") == 1
    assert "tela" not in result.description
    assert result.scene_context["objects"] == ["computer", "screen"]


def test_separate_screen_is_preserved(narrative):
    item = scene(narrative)
    item.context["raw_outputs"]["object"]["crop_results"].append(
        {"detector_label": "tv", "mapped_labels": ["screen"]})
    assert "tela" in narrative.generator.generate(item).description


@pytest.mark.parametrize("label, phrase", [("writing", "há quem escreva"),
                                          ("computer_use", "há quem use computador"),
                                          ("working", "há quem realize uma atividade")])
def test_confirmed_activity_overrides_posture_without_universalizing(narrative, label, phrase):
    item = scene(narrative)
    confirm(item, label)
    result = narrative.generator.generate(item)
    assert result.scene_context["action"] == label
    assert phrase in result.description
    assert "todos" not in result.description and "trabalham" not in result.description


def test_conflicting_global_actions_do_not_choose_an_activity(narrative):
    item = scene(narrative)
    item.labels += ["writing", "cooking"]
    item.confidence.update({"action.writing": .98, "action.cooking": .99})
    result = narrative.generator.generate(item)
    assert result.scene_context["action"] is None
    assert "escreva" not in result.description and "cozinhe" not in result.description


def test_confirmed_preference_wins_over_conflicting_global_scores(narrative):
    item = scene(narrative)
    confirm(item)
    item.labels.append("cooking")
    item.confidence["action.cooking"] = .99
    assert narrative.generator.generate(item).scene_context["action"] == "writing"


def test_group_with_only_postures(narrative):
    item = scene(narrative, objects=False)
    item.labels = ["person", "indoor", "still", "sitting", "standing"]
    item.confidence.update({"action.sitting": .9, "action.standing": .8})
    assert narrative.generator.generate(item).scene_context["action"] is None


def test_confirmed_posture_does_not_become_an_activity(narrative):
    item = scene(narrative)
    confirm(item, "sitting")
    result = narrative.generator.generate(item)
    assert result.scene_context["action"] == "sitting"
    assert "há quem esteja sentado" in result.description
    assert "use computador" not in result.description and "trabalh" not in result.description


@pytest.mark.parametrize("change", ["inconclusive", "missing_label", "low_score",
                                    "no_person_support", "wrong_source", "wrong_evidence_label"])
def test_unconfirmed_activity_is_not_inferred(narrative, change):
    item = scene(narrative)
    confirm(item, "computer_use")
    evidence = item.context["preferred_action_evidence"]
    if change == "inconclusive":
        evidence["evidence_passed"] = False
    elif change == "missing_label":
        item.labels.remove("computer_use")
    elif change == "low_score":
        item.confidence["action.computer_use"] = .59
    elif change == "no_person_support":
        evidence["person_supported_clips"] = 0
    elif change == "wrong_source":
        item.context["preferred_action_source"] = "frame_action"
    else:
        evidence["label"] = "writing"
    result = narrative.generator.generate(item)
    assert result.scene_context["action"] is None
    assert "use computador" not in result.description


def test_single_person_keeps_previous_behavior(narrative):
    item = scene(narrative)
    item.context["person_intermediate"].update(count=1, count_type="exact")
    result = narrative.generator.generate(item)
    assert result.scene_context["action"] == "still"
    assert result.description == "Uma pessoa permanece parada em um ambiente interno."


def test_without_intermediate_matches_original_fusion(narrative):
    item = scene(narrative)
    item.context.pop("person_intermediate")
    assert narrative.generator.generate(item) == narrative.base.generate(item)


def test_car_driving_priority_and_vehicle_focus(narrative):
    item = narrative.Input(labels=["car", "road", "still", "driving"],
                           confidence={"object.car": .9468, "scene.road": .6836,
                                       "action.still": .927, "action.driving": .65},
                           context={"preferred_action": "driving",
                                    "vehicle_action_hint": {"action": "driving"}})
    # Even stale person metadata must not override an established vehicle focus.
    item.context["person_intermediate"] = {"status": "analyzed", "count": 3, "count_type": "group"}
    result = narrative.generator.generate(item)
    assert result.scene_context["action"] == "driving"
    assert result.scene_context["vehicle_focus"] == "car"
    assert result.scene_context["subject"] is None
    assert "carro" in result.description and "movimento" in result.description
    assert "pessoa" not in result.description and "motorista" not in result.description


def test_suppressed_false_animal_is_not_verbalized(narrative):
    item = scene(narrative, objects=False)
    item.labels.append("animal")
    item.confidence["object.animal"] = .95
    item.context["suppressed_object_labels"] = ["animal"]
    result = narrative.generator.generate(item)
    assert "animal" not in result.scene_context["objects"]
    assert "animal" not in result.description


def test_continuity_skips_identical_facts_but_keeps_new_objects_and_activity(narrative):
    first = scene(narrative, objects=False)
    repeated = copy.deepcopy(first)
    repeated.start_time, repeated.end_time = 5., 10.
    with_object = scene(narrative)
    with_object.start_time, with_object.end_time = 10., 15.
    with_activity = copy.deepcopy(with_object)
    with_activity.start_time, with_activity.end_time = 15., 20.
    confirm(with_activity)
    outputs = narrative.generator.generate_batch([first, repeated, with_object, with_activity])
    assert [output.skipped for output in outputs] == [False, True, False, False]
    assert "computador" in outputs[2].description
    assert "há quem escreva" in outputs[3].description


def test_continuity_ignores_detection_metadata_changes(narrative):
    item = scene(narrative)
    first = narrative.generator.generate(item).scene_context
    item.context["person_intermediate"].update(frame_counts=[3, 3], representative_timestamp=8.)
    item.confidence["object.computer"] = .8
    current = narrative.generator.generate(item).scene_context
    assert narrative.generator._build_delta_description(current, first) == ""


@pytest.mark.parametrize("indoor", [True, False])
def test_school_is_not_asserted_as_institution_or_replaced_by_office(narrative, indoor):
    item = scene(narrative)
    item.labels.append("school")
    item.confidence["scene.school"] = .8428
    if not indoor:
        item.labels.remove("indoor")
    result = narrative.generator.generate(item)
    assert result.scene_context["environment"] == ("indoor" if indoor else None)
    assert "escola" not in result.description and "escritório" not in result.description


def test_llm_guard_still_rejects_new_facts_and_keeps_prompt_constraints(narrative):
    generator = narrative.generator
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return "Um grupo de pessoas trabalha em um escritório."

    generator.use_llm = True
    generator.client = SimpleNamespace(generate=fake_generate)
    base = generator.generate(scene(narrative)).description
    assert generator._safe_llm_rewrite(base) == base
    assert "Preserve todos os fatos existentes" in prompts[0]
    assert "Não acrescente sujeitos, ações" in prompts[0]


@pytest.mark.parametrize("job_id", ["9a7d3408c192465895ee50de553c2331",
                                    "badda904e2c44d01991f249b1a700eb0"])
def test_saved_job_facts_without_inference(narrative, job_id):
    path = ROOT / "data/output/api/generations" / job_id / "spectra/spectra_outputs.json"
    if not path.is_file():
        pytest.skip("Optional local job artifact unavailable")
    saved = json.loads(path.read_text(encoding="utf-8"))[0]
    item = narrative.Input(**{key: saved[key] for key in
                              ("labels", "confidence", "context", "start_time", "end_time")})
    original = copy.deepcopy(item)
    result = narrative.generator.generate(item)
    if job_id.startswith("9a7"):
        assert result.scene_context["objects"] == ["computer", "screen"]
        assert result.scene_context["action"] is None
        assert "computador" in result.description and "tela" not in result.description
        assert "escola" not in result.description and "escritório" not in result.description
        print("Office base:", result.description)
    else:
        assert result.scene_context["action"] == "driving"
        assert result.scene_context["vehicle_focus"] == "car"
        assert "animal" not in result.description
        assert result == narrative.base.generate(item)
    assert item == original
