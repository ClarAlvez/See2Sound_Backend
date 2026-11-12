"""Progress uses temporary job records and mocked models/media only."""
import importlib.util
import itertools
import json
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from tests.test_api_stability import api_modules


def status_client(api):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(api.routes.router)
    return TestClient(app)


def test_status_progress_and_existing_fields(api_modules):
    api = api_modules
    api.store.update_job("testjob", status="processing", progress_percent=65,
                         stage="spectra", stage_label="Analisando cenas",
                         stage_durations_seconds={"metadata": 0.1})
    with status_client(api) as client:
        response = client.get("/api/v1/generations/testjob")
    assert response.status_code == 200
    body = response.json()
    assert body["progress_percent"] == 65
    assert body["stage"] == "spectra" and body["stage_label"] == "Analisando cenas"
    assert body["stage_current"] is None and body["stage_total"] is None
    assert body["job_id"] == "testjob" and body["options"] == {}
    assert body["stage_durations_seconds"] == {"metadata": 0.1}
    assert {"created_at", "started_at", "finished_at", "error", "original_filename"} <= body.keys()


@pytest.mark.parametrize("state, expected", [
    ("queued", 0), ("processing", 0), ("failed", 0), ("completed", 100),
])
def test_old_records_remain_readable(api_modules, state, expected):
    api = api_modules
    job = api.store.read_job("testjob")
    for key in ("progress_percent", "stage", "stage_label", "stage_current", "stage_total"):
        job.pop(key, None)
    job["status"] = state
    api.store.get_job_path("testjob").write_text(json.dumps(job), encoding="utf-8")
    with status_client(api) as client:
        response = client.get("/api/v1/generations/testjob")
    assert response.status_code == 200
    assert response.json()["progress_percent"] == expected
    assert response.json()["stage_label"] is None


@pytest.mark.parametrize("fail", [False, True])
def test_simulated_generation_completion_and_failure(api_modules, fail):
    api = api_modules
    snapshots = []

    def pipeline(**kwargs):
        for stage in ("metadata", "speech_analysis", "spectra", "initial_artifacts",
                      "xclip", "verification_processing", "narrative", "voice_engine"):
            kwargs["progress_callback"](stage)
            snapshots.append(api.store.read_job("testjob"))
        if fail:
            raise RuntimeError("simulated failure")
        return {"artifacts": {}, "unchanged": True}

    api.pipeline.process_video_verified.side_effect = pipeline
    api.service.run_generation_job("testjob", "unused", {})
    percentages = [job["progress_percent"] for job in snapshots]
    assert percentages == sorted(percentages) and len(set(percentages)) > 3
    assert all(job["status"] == "processing" and 0 <= job["progress_percent"] <= 99
               for job in snapshots)
    final = api.store.read_job("testjob")
    if fail:
        assert final["status"] == "failed" and final["progress_percent"] == percentages[-1]
        assert final["stage"] == "voice_engine"
    else:
        assert final["status"] == "completed" and final["progress_percent"] == 100
        assert json.loads(Path(final["result_path"]).read_text()) == {"artifacts": {}, "unchanged": True}
    with status_client(api) as client:
        assert client.get("/api/v1/generations/testjob").json()["progress_percent"] == final["progress_percent"]


@pytest.mark.parametrize("spectra,narrative,tts", list(itertools.product([False, True], repeat=3)))
def test_optional_stages_have_no_weight_or_updates(api_modules, monkeypatch, spectra, narrative, tts):
    api = api_modules
    api.store.update_job("testjob", status="processing")
    write = Mock(wraps=api.service.update_job)
    monkeypatch.setattr(api.service, "update_job", write)
    callback = api.service._progress_callback("testjob", {
        "run_spectra": spectra, "run_narrative": narrative, "run_tts": tts,
    })
    for stage in ("metadata", "audio_extraction", "speech_analysis", "frames", "scenes",
                  "spectra", "initial_artifacts", "xclip", "verification_processing",
                  "verified_artifacts", "narrative", "voice_engine", "saving_artifacts",
                  "extended_playback", "saving_result"):
        callback(stage)
    stages = [call.kwargs["stage"] for call in write.call_args_list]
    assert ("spectra" in stages) == spectra
    assert ("xclip" in stages) == spectra
    assert ("narrative" in stages) == (spectra and narrative)
    assert ("voice_engine" in stages) == (spectra and narrative and tts)
    weights = 30 + (52 if spectra else 0) + (15 if spectra and narrative else 0)
    weights += 15 if spectra and narrative and tts else 0
    weights += 7  # artifact finalization, extended video and result save
    assert api.store.read_job("testjob")["progress_percent"] == int(99 * (weights - 1) / weights)
    count = write.call_count
    callback("saving_result")
    callback("metadata")
    assert write.call_count == count


def test_store_monotonic_bounded_and_isolated(api_modules):
    store = api_modules.store
    store.create_job("other", "other.mp4", "unused", 1, {})
    assert store.read_job("other")["progress_percent"] == 0
    store.update_job("testjob", status="processing", progress_percent=65)
    assert store.update_job("testjob", progress_percent=20)["progress_percent"] == 65
    assert store.update_job("testjob", progress_percent=200)["progress_percent"] == 99
    assert store.update_job("testjob", status="failed", progress_percent=100)["progress_percent"] == 99
    assert store.read_job("other")["progress_percent"] == 0


def test_progress_write_failure_does_not_fail_generation(api_modules, monkeypatch):
    api = api_modules
    real_update = api.store.update_job

    def update(job_id, **changes):
        if "stage" in changes:
            raise OSError("progress storage unavailable")
        return real_update(job_id, **changes)

    monkeypatch.setattr(api.service, "update_job", update)
    api.pipeline.process_video_verified.side_effect = lambda **kw: (
        kw["progress_callback"]("spectra") or {"artifacts": {}})
    api.service.run_generation_job("testjob", "unused", {})
    assert api.store.read_job("testjob")["status"] == "completed"


@pytest.fixture
def verified_pipeline(monkeypatch):
    # Stub imports before loading the real orchestration; no AI module is executed.
    imports = {
        "ai.spectra.Object.object_intermediate": ["ObjectCLIPVerifier", "apply_object_intermediate", "scene_for_narrative"],
        "ai.narrative.person_fusion_narrative": ["PersonFusionNarrativeGenerator"],
        "ai.spectra.Person.person_intermediate": ["PersonCLIPVerifier", "apply_person_intermediate"],
        "ai.spectra.Actions.xclip_verifier": ["get_xclip_verifier"],
        "ai.spectra.Actions.action_comparator": ["compare_scene", "build_summary"],
        "ai.spectra.Actions.action_fusion": ["apply_action_correction"],
        "ai.narrative.fusion_narrative": ["FusionNarrativeGenerator"],
        "pipeline.orchestration.process_video": ["process_video", "generate_voice_audio", "save_json"],
        "cv2": [],
        "ai.voice_engine.extended_playback": ["extend_video_when_needed"],
    }
    for name, attrs in imports.items():
        module = ModuleType(name)
        for attr in attrs:
            setattr(module, attr, Mock())
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "pipeline/orchestration/verified_pipeline.py"
    spec = importlib.util.spec_from_file_location("progress_verified_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.save_json.side_effect = lambda **kw: kw["output_path"]
    module.run_xclip_comparison = Mock(return_value={"scenes": [{"scene": 1}], "summary": {}})
    module.scene_for_narrative.side_effect = lambda scene: scene
    module.PersonFusionNarrativeGenerator.return_value.generate_timeline_from_dicts.return_value = [{"description": "test"}]
    module.generate_voice_audio.return_value = {"modified_audio_path": "audio.wav"}
    monkeypatch.setenv("SEE2SOUND_PERSON_INTERMEDIATE", "false")
    monkeypatch.setenv("SEE2SOUND_OBJECT_INTERMEDIATE", "false")
    return module


def test_verified_progress_continues_after_initial_pipeline(api_modules, verified_pipeline, monkeypatch):
    api, pipeline = api_modules, verified_pipeline
    snapshots = []

    def initial(**kwargs):
        assert not kwargs["run_narrative"] and not kwargs["run_tts"]
        for stage in ("metadata", "spectra", "narrative", "voice_engine", "saving_artifacts"):
            kwargs["progress_callback"](stage)
        job = api.store.read_job("testjob")
        assert job["stage"] == "initial_artifacts" and job["progress_percent"] < 99
        return {"spectra": {"outputs": [{"scene": 1}]}, "metadata": {},
                "audio": {"extraction": {}, "speech_analysis": {}}}

    pipeline.process_video.side_effect = initial
    real_update = api.store.update_job

    def update(job_id, **changes):
        job = real_update(job_id, **changes)
        snapshots.append(job)
        return job

    monkeypatch.setattr(api.service, "update_job", update)
    api.pipeline.process_video_verified.side_effect = pipeline.process_video_verified
    api.service.run_generation_job("testjob", "unused", {})
    stages = [job.get("stage") for job in snapshots]
    assert stages[1:-1] == ["metadata", "spectra", "initial_artifacts", "xclip",
                           "verification_processing", "verified_artifacts", "narrative",
                           "voice_engine", "saving_artifacts", "extended_playback", "saving_result"]
    assert all(job["progress_percent"] < 100 for job in snapshots[:-1])
    assert snapshots[-1]["status"] == "completed" and snapshots[-1]["progress_percent"] == 100
    result = json.loads(Path(snapshots[-1]["result_path"]).read_text())
    assert result["narrative"]["timeline"] == [{"description": "test"}]
    assert result["voice_engine"] == {"modified_audio_path": "audio.wav"}
    assert "progress_percent" not in result


@pytest.mark.parametrize("spectra,xclip", [(False, True), (True, False)])
def test_traditional_pipeline_callback_forwarded(verified_pipeline, spectra, xclip):
    pipeline = verified_pipeline
    callback = Mock(side_effect=OSError("secondary callback failure"))

    def original(**kw):
        kw["progress_callback"]("metadata")
        return {"unchanged": True}

    pipeline.process_video.side_effect = original
    assert pipeline.process_video_verified("unused", run_spectra=spectra, run_xclip=xclip,
                                           progress_callback=callback) == {"unchanged": True}
    callback.assert_called_once_with("metadata")
    pipeline.run_xclip_comparison.assert_not_called()


@pytest.mark.parametrize("narrative,tts", list(itertools.product([False, True], repeat=2)))
def test_verified_optional_stages_preserve_result(verified_pipeline, narrative, tts):
    pipeline = verified_pipeline
    original = {"spectra": {"outputs": [{"scene": 1}]}, "metadata": {},
                "audio": {"extraction": {}, "speech_analysis": {}}}

    def initial(**kw):
        callback = kw["progress_callback"]
        for stage in ("spectra", "narrative", "voice_engine", "saving_artifacts"):
            callback(stage)
        return deepcopy(original)

    pipeline.process_video.side_effect = initial
    baseline = pipeline.process_video_verified("unused", run_narrative=narrative, run_tts=tts)
    events = []
    result = pipeline.process_video_verified("unused", run_narrative=narrative,
                                             run_tts=tts, progress_callback=events.append)
    assert result == baseline
    assert events[:2] == ["spectra", "initial_artifacts"]
    assert ("narrative" in events) == narrative
    assert ("voice_engine" in events) == (narrative and tts)
    assert events[-1] == "saving_artifacts"
