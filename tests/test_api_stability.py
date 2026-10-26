"""API failure handling with real temporary job storage and a mocked pipeline."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api_modules(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        jobs_dir=tmp_path / "jobs", generations_dir=tmp_path / "generations",
        max_pending_generations=4, validate_model_files=Mock(),
        **{name + "_model_path": tmp_path / (name + ".model")
           for name in ("scene", "person", "object", "atmosphere", "action", "narrative")},
    )
    settings.jobs_dir.mkdir()
    config = ModuleType("API.core.config")
    config.settings = settings
    monkeypatch.setitem(sys.modules, "API.core.config", config)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    store = load("API.services.job_store", "API/services/job_store.py")
    service = load("API.services.generation_service", "API/services/generation_service.py")
    runtime = load("API.services.runtime", "API/services/runtime.py")
    load("API.utils.video_validation", "API/utils/video_validation.py")
    routes = load("API.routes.generations", "API/routes/generations.py")
    pipeline = ModuleType("pipeline.orchestration.verified_pipeline")
    pipeline.process_video_verified = Mock(return_value={"artifacts": {}, "unchanged": True})
    monkeypatch.setitem(sys.modules, pipeline.__name__, pipeline)
    store.create_job("testjob", "video.mp4", "unused", 1, {})
    return SimpleNamespace(settings=settings, store=store, service=service,
                           runtime=runtime, routes=routes, pipeline=pipeline)


def test_directory_failure_marks_job_failed(api_modules, caplog):
    api = api_modules
    api.settings.generations_dir.mkdir()
    # A file where the per-job directory must be created: deterministic on Windows/Linux.
    (api.settings.generations_dir / "testjob").write_text("keep", encoding="utf-8")
    api.service.run_generation_job("testjob", "unused", {})
    job = api.store.read_job("testjob")
    assert job["status"] == "failed"
    assert job["finished_at"] and "FileExistsError" in job["error"]
    api.pipeline.process_video_verified.assert_not_called()
    assert "job=testjob generation_failed" in caplog.text
    assert (api.settings.generations_dir / "testjob").read_text() == "keep"


def test_failure_to_persist_failed_state_preserves_original(api_modules, monkeypatch, caplog):
    api = api_modules
    original = PermissionError("original storage error")
    secondary = OSError("secondary storage error")
    monkeypatch.setattr(api.service, "update_job", Mock(side_effect=[original, secondary]))
    with pytest.raises(PermissionError) as raised:
        api.service.run_generation_job("testjob", "unused", {})
    assert raised.value is original
    assert "generation_failed" in caplog.text
    assert "failed_state_persistence_failed" in caplog.text
    assert "original storage error" in caplog.text and "secondary storage error" in caplog.text
    api.pipeline.process_video_verified.assert_not_called()


def test_initial_status_failure_can_still_be_recorded(api_modules, monkeypatch):
    api = api_modules
    real_update = api.store.update_job

    def update(job_id, **changes):
        if changes["status"] == "processing":
            raise OSError("initial write failed")
        return real_update(job_id, **changes)

    monkeypatch.setattr(api.service, "update_job", update)
    api.service.run_generation_job("testjob", "unused", {})
    job = api.store.read_job("testjob")
    assert job["status"] == "failed" and "initial write failed" in job["error"]


def test_pipeline_error_is_recorded_without_changing_inference(api_modules):
    api = api_modules
    api.pipeline.process_video_verified.side_effect = ValueError("mocked inference failure")
    api.service.run_generation_job("testjob", "unused", {})
    job = api.store.read_job("testjob")
    assert job["status"] == "failed" and job["error"] == "ValueError: mocked inference failure"


def test_success_persists_original_result_and_options(api_modules):
    api = api_modules
    api.service.run_generation_job("testjob", "unused", {"frame_interval_seconds": 2.0})
    job = api.store.read_job("testjob")
    assert job["status"] == "completed"
    assert json.loads(Path(job["result_path"]).read_text()) == {"artifacts": {}, "unchanged": True}
    kwargs = api.pipeline.process_video_verified.call_args.kwargs
    assert kwargs["frame_interval_seconds"] == 2.0
    assert kwargs["video_path"] == "unused" and kwargs["run_spectra"] is True
    assert kwargs["run_narrative"] is True and kwargs["run_tts"] is True


def test_serial_lock_and_slot_release_survive_worker_failure(api_modules):
    runtime = api_modules.runtime
    seen = []
    assert runtime.reserve_generation_slot()

    def worker(*args):
        seen.append(runtime._pipeline_lock.locked())
        raise OSError("worker failed")

    with pytest.raises(OSError):
        runtime.run_admitted_generation("testjob", "unused", {}, worker)
    assert seen == [True] and not runtime._pipeline_lock.locked()
    assert all(runtime.reserve_generation_slot() for _ in range(4))
    assert not runtime.reserve_generation_slot()
    for _ in range(4):
        runtime.release_generation_slot()


@pytest.mark.parametrize("state", ["queued", "processing", "failed", "completed"])
def test_artifact_http_states(api_modules, monkeypatch, tmp_path, state):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    api = api_modules
    api.store.update_job("testjob", status=state, result_path="unused")
    artifact = api.settings.generations_dir / "testjob" / "artifact.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("original artifact", encoding="utf-8")
    loader = Mock(return_value={"artifacts": {"sample": str(artifact)}})
    monkeypatch.setattr(api.routes, "load_generation_result", loader)
    app = FastAPI()
    app.include_router(api.routes.router)
    with TestClient(app) as client:
        response = client.get("/api/v1/generations/testjob/artifacts/sample")
    if state == "completed":
        assert response.status_code == 200 and response.text == "original artifact"
    else:
        assert response.status_code == 409
        loader.assert_not_called()
        expected = ("A geração falhou. Consulte o status da geração para detalhes."
                    if state == "failed" else "A geração ainda não terminou.")
        assert response.json() == {"detail": expected}
