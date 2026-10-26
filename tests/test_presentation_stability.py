"""Final peripheral regression: only mocks, temporary files and local sockets."""
import asyncio
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.test_api_stability import api_modules

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_upload_cleanup_and_close_do_not_mask_original(tmp_path, monkeypatch, caplog):
    files = load_module("presentation_uploads", "API/utils/files.py", monkeypatch)
    original = OSError("original read failure")
    upload = SimpleNamespace(file=Mock(read=Mock(side_effect=original)),
                             close=AsyncMock(side_effect=OSError("close failure")))
    real_unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path == tmp_path / "upload.mp4":
            raise PermissionError("cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", denied)
    with pytest.raises(OSError) as raised:
        asyncio.run(files.save_upload_file(upload, tmp_path / "upload.mp4", 100))
    assert raised.value is original
    assert "cleanup failure" in caplog.text and "close failure" in caplog.text


def test_normal_upload_keeps_original_bytes(tmp_path, monkeypatch):
    files = load_module("presentation_uploads", "API/utils/files.py", monkeypatch)
    data = b"unchanged uploaded bytes"
    upload = SimpleNamespace(file=io.BytesIO(data), close=AsyncMock())
    path = tmp_path / "upload.mp4"
    assert asyncio.run(files.save_upload_file(upload, path, 100)) == len(data)
    assert path.read_bytes() == data
    upload.close.assert_awaited_once()


def test_oversize_upload_removes_only_its_partial_file(tmp_path, monkeypatch):
    files = load_module("presentation_uploads", "API/utils/files.py", monkeypatch)
    sentinel = tmp_path / "other-result.wav"
    sentinel.write_bytes(b"preserve")
    upload = SimpleNamespace(file=io.BytesIO(b"too large"), close=AsyncMock())
    path = tmp_path / "upload.mp4"
    with pytest.raises(files.UploadTooLargeError):
        asyncio.run(files.save_upload_file(upload, path, 1))
    assert not path.exists() and sentinel.read_bytes() == b"preserve"


def test_upload_rejection_releases_slot_even_when_close_fails(api_modules, tmp_path, monkeypatch):
    from fastapi import BackgroundTasks, HTTPException

    api = api_modules
    api.settings.upload_dir = tmp_path / "uploads"
    api.settings.allowed_video_extensions = {".mp4"}
    api.settings.max_upload_size_bytes = 1000
    upload = SimpleNamespace(filename="sample.mp4", file=io.BytesIO(b"mock video"),
                             close=AsyncMock(side_effect=OSError("close failed")))
    monkeypatch.setattr(api.routes, "validate_uploaded_video",
                        Mock(side_effect=api.routes.InvalidVideoError("invalid mock video")))
    tasks = BackgroundTasks()
    with pytest.raises(HTTPException) as raised:
        asyncio.run(api.routes.create_generation(tasks, upload, 1.0, None, True, True, True))
    assert raised.value.status_code == 400
    assert not tasks.tasks
    assert not list(api.settings.upload_dir.iterdir())
    assert all(api.runtime.reserve_generation_slot() for _ in range(4))
    assert not api.runtime.reserve_generation_slot()
    for _ in range(4):
        api.runtime.release_generation_slot()


def test_job_temp_cleanup_does_not_mask_write_failure(api_modules, monkeypatch, caplog):
    api = api_modules
    original = OSError("original JSON write failure")
    monkeypatch.setattr(api.store.json, "dump", Mock(side_effect=original))
    real_unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise PermissionError("temporary cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", denied)
    with pytest.raises(OSError) as raised:
        api.store.update_job("testjob", status="failed")
    assert raised.value is original
    assert api.store.read_job("testjob")["status"] == "queued"
    assert "temporary cleanup failure" in caplog.text


@pytest.mark.parametrize("content", ["{truncated", "[]", '{"artifacts": []}', '{"voice_engine": []}'])
def test_unreadable_result_uses_safe_stored_error(api_modules, tmp_path, content):
    from API.core.errors import StoredDataError

    path = tmp_path / "result.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(StoredDataError) as raised:
        api_modules.service.load_generation_result("testjob", str(path))
    assert str(path) not in str(raised.value)
    assert path.read_text(encoding="utf-8") == content


def application(api, tmp_path, monkeypatch):
    api.settings.cors_origins = ["*"]
    api.settings.upload_dir = tmp_path / "uploads"

    def directories():
        for path in (api.settings.upload_dir, api.settings.jobs_dir, api.settings.generations_dir):
            path.mkdir(parents=True, exist_ok=True)

    api.settings.create_runtime_directories = directories
    return load_module("presentation_application", "API/main.py", monkeypatch).app


@pytest.mark.parametrize("kind", ["missing", "corrupt", "valid"])
def test_real_app_startup_and_result_routes(api_modules, tmp_path, monkeypatch, kind):
    from fastapi.testclient import TestClient

    api = api_modules
    path = tmp_path / "result.json"
    content = {"artifacts": {}, "voice_engine": None, "narrative": {"timeline": []}}
    if kind != "missing":
        path.write_text(json.dumps(content) if kind == "valid" else "{broken", encoding="utf-8")
    api.store.update_job("testjob", status="completed", result_path=str(path))
    heavy = {name for name in sys.modules if name.split(".")[0] in {"torch", "transformers", "cv2", "ultralytics", "llama_cpp"}}
    with TestClient(application(api, tmp_path, monkeypatch)) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/").json()["service"] == "See2Sound API"
        for route in ("result", "audio-description", "audio-description/metadata", "artifacts/sample"):
            response = client.get("/api/v1/generations/testjob/" + route)
            if kind == "valid":
                assert response.status_code == (200 if route == "result" else 404)
                if route == "result":
                    assert response.json()["result"] == content
            else:
                assert response.status_code == 500
                assert set(response.json()) == {"detail"}
                assert str(tmp_path) not in response.text and "Traceback" not in response.text
    assert heavy == {name for name in sys.modules if name.split(".")[0] in {"torch", "transformers", "cv2", "ultralytics", "llama_cpp"}}
    assert api.store.read_job("testjob")["status"] == "completed"


def test_app_recovers_only_temporary_interrupted_job(api_modules, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    with TestClient(application(api_modules, tmp_path, monkeypatch)) as client:
        job = client.get("/api/v1/generations/testjob").json()
    assert job["status"] == "failed" and job["stage"] == "interrupted"


def test_port_preflight_reports_occupied_local_port(monkeypatch, capsys):
    diagnostic = load_module("presentation_diagnostic", "tools/check_backend.py", monkeypatch)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        assert diagnostic.port_available(server.getsockname()[1]) is False
        # Drain the first probe before the CLI probe; otherwise Windows may time
        # out on the deliberately unconsumed backlog rather than report a listener.
        server.settimeout(2)
        with server.accept()[0]:
            pass
        monkeypatch.setattr(diagnostic, "load_settings", Mock(side_effect=AssertionError("Must stop before config")))
        assert diagnostic.main(["--skip-api", "--require-port-free", "--api-port", str(server.getsockname()[1])]) == 1
    assert "ocupada" in capsys.readouterr().out


def test_port_preflight_handles_permission_error(monkeypatch, capsys):
    diagnostic = load_module("presentation_diagnostic", "tools/check_backend.py", monkeypatch)
    monkeypatch.setattr(diagnostic, "port_available", Mock(side_effect=PermissionError("socket denied")))
    assert diagnostic.main(["--skip-api", "--require-port-free"]) == 1
    assert "socket denied" in capsys.readouterr().out


def test_runtime_lease_rejects_second_owner_and_releases_on_error(api_modules):
    runtime = api_modules.runtime
    with pytest.raises(ValueError, match="simulated startup failure"):
        with runtime.RuntimeLease():
            with pytest.raises(RuntimeError, match="Outra API"):
                with runtime.RuntimeLease():
                    pytest.fail("Second owner unexpectedly admitted")
            raise ValueError("simulated startup failure")
    with runtime.RuntimeLease():
        pass
