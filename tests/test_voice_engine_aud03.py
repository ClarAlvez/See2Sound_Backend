"""AUD-03: mocked voice operations, synthetic WAVs and temporary job storage only."""
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import wave

import pytest

from tests.test_api_stability import api_modules

ROOT = Path(__file__).resolve().parents[1]
NO_PAUSE_REASON = "Nenhum espaço de áudio suficientemente longo foi encontrado."


def write_wav(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x00" * int(seconds * 44100))
    return str(path)


@pytest.fixture
def voice(tmp_path, monkeypatch):
    models = importlib.import_module("ai.voice_engine.data_models")
    tts, scheduler, mixer = Mock(), Mock(), Mock()
    scheduler.build_available_pauses.return_value = [{"start": 0.0, "end": 12.0, "duration": 12.0}]

    def slot(**kwargs):
        start = 1.0 + (scheduler.find_slot.call_count - 1) * 4
        return {"pause_start": 0.0, "pause_end": 12.0,
                "playback_start": start, "playback_end": start + 2.5,
                "occupied_start": start - .1, "occupied_end": start + 2.6}

    scheduler.find_slot.side_effect = slot
    tts.save_to_file.side_effect = lambda text, output_path: write_wav(Path(output_path), 2.5)
    mixer.get_duration.return_value = 2.5

    def mix(**kwargs):
        # A fake mixer preserves the input bytes; no DSP or real voice synthesis.
        shutil.copyfile(kwargs["original_audio_path"], kwargs["output_path"])
        return str(Path(kwargs["output_path"]).resolve())

    mixer.mix.side_effect = mix
    package = ModuleType("ai.audio_description")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name, cls, value in (
        ("ai.audio_description.tts_client", "TTSClient", tts),
        ("ai.voice_engine.pause_scheduler", "PauseScheduler", scheduler),
        ("ai.voice_engine.audio_mixer", "VoiceAudioMixer", mixer),
    ):
        module = ModuleType(name)
        setattr(module, cls, Mock(return_value=value))
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("aud03_voice_engine", ROOT / "ai/voice_engine/voice_engine.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    original = write_wav(tmp_path / "original.wav", 12)

    def build(output_dir):
        return module.VoiceEngine(str(output_dir), background_volume=.18, description_volume=1.0)

    engine = build(tmp_path / "voice")
    return SimpleNamespace(models=models, module=module, engine=engine, build=build,
                           tts=tts, scheduler=scheduler, mixer=mixer, original=original)


def timeline(count=1):
    return [{"description": "Um carro segue pela rua.", "start_time": 0.0, "end_time": 12.0}
            for _ in range(count)]


def generate(voice, items=None):
    return voice.engine.generate(voice.original, timeline() if items is None else items, [], 12.0)


@pytest.mark.parametrize("count", [1, 2])
def test_success_counts_artifacts_and_legacy_fields(voice, count):
    result = generate(voice, timeline(count))
    assert result.total_descriptions == result.inserted_descriptions == count
    assert result.skipped_descriptions == 0
    assert result.diagnostics["outcome"] == "success"
    assert result.diagnostics["technical_failures"] == 0
    assert all(cue.skip_reason is None and cue.outcome == "inserted" for cue in result.cues)
    assert Path(result.modified_audio_path).read_bytes() == Path(voice.original).read_bytes()
    with wave.open(result.modified_audio_path) as stream:
        assert stream.getnframes() / stream.getframerate() == 12.0
    assert json.loads(Path(result.manifest_path).read_text(encoding="utf-8")) == result.to_dict()
    assert set(result.cues[0].to_dict()) == {
        "index", "text", "scene_start_time", "scene_end_time", "pause_start", "pause_end",
        "playback_start", "playback_end", "tts_duration", "audio_path", "inserted", "skip_reason",
    }
    assert result.cues[0].tts_duration == 2.5
    assert (result.cues[0].playback_start, result.cues[0].playback_end) == (1.0, 3.5)
    assert voice.mixer.mix.call_args.kwargs["background_volume"] == .18
    assert voice.mixer.mix.call_args.kwargs["description_volume"] == 1.0


@pytest.mark.parametrize("cause", [ConnectionError("provider unavailable"),
                                        RuntimeError("conversion failed"),
                                        RuntimeError("invalid WAV from TTS")])
def test_all_tts_operations_fail_explicitly(voice, cause, caplog):
    voice.tts.save_to_file.side_effect = cause
    with pytest.raises(voice.models.VoiceGenerationError) as raised:
        generate(voice, timeline(2))
    assert raised.value.__cause__ is cause
    assert raised.value.diagnostics["outcome"] == "technical_failure"
    assert raised.value.diagnostics["tts_generation_failures"] == 2
    assert raised.value.diagnostics["technical_failures"] == 2
    assert raised.value.diagnostics["no_pause_descriptions"] == 0
    assert str(cause) in caplog.text and "stage=tts_generation" in caplog.text
    voice.mixer.mix.assert_not_called()
    assert not (voice.engine.output_dir / "audio_with_description.wav").exists()


def test_partial_success_keeps_audio_and_sanitizes_skip_reason(voice, caplog):
    cause = ConnectionError(r"C:\private\file token=SIMULATED_SECRET")
    synthesize = voice.tts.save_to_file.side_effect

    def fail_first(**kwargs):
        if voice.tts.save_to_file.call_count == 1:
            raise cause
        return synthesize(**kwargs)

    voice.tts.save_to_file.side_effect = fail_first
    result = generate(voice, timeline(2))
    assert (result.total_descriptions, result.inserted_descriptions, result.skipped_descriptions) == (2, 1, 1)
    assert result.diagnostics["outcome"] == "partial_success"
    assert result.diagnostics["tts_generation_failures"] == 1
    assert result.diagnostics["no_pause_descriptions"] == 0
    assert result.cues[0].skip_reason and "SIMULATED_SECRET" not in json.dumps(result.to_dict())
    assert "SIMULATED_SECRET" in caplog.text
    assert result.cues[1].playback_start == 1.0
    assert Path(result.modified_audio_path).is_file()


def test_no_pause_is_not_a_technical_failure(voice):
    voice.scheduler.find_slot.side_effect = None
    voice.scheduler.find_slot.return_value = None
    result = generate(voice)
    assert result.inserted_descriptions == 0 and result.skipped_descriptions == 1
    assert result.diagnostics["outcome"] == "no_available_pause"
    assert result.diagnostics["technical_failures"] == 0
    assert result.diagnostics["no_pause_descriptions"] == 1
    assert result.cues[0].skip_reason == NO_PAUSE_REASON
    assert Path(result.modified_audio_path).is_file()


@pytest.mark.parametrize("failure_kind", ["no_pause", "audio_validation", "scheduling"])
def test_partial_non_provider_failure_preserves_inserted_description(voice, failure_kind):
    slot = voice.scheduler.find_slot.side_effect
    if failure_kind == "audio_validation":
        voice.mixer.get_duration.side_effect = [2.5, ValueError("invalid second audio")]
    else:
        def second_attempt(**kwargs):
            if voice.scheduler.find_slot.call_count == 2:
                if failure_kind == "scheduling":
                    raise RuntimeError("second scheduling failed")
                return None
            return slot(**kwargs)
        voice.scheduler.find_slot.side_effect = second_attempt
    result = generate(voice, timeline(2))
    assert result.diagnostics["outcome"] == "partial_success"
    assert result.inserted_descriptions == 1 and result.skipped_descriptions == 1
    assert result.diagnostics["tts_generation_failures"] == 0
    assert result.diagnostics["technical_failures"] == (0 if failure_kind == "no_pause" else 1)
    assert result.diagnostics["no_pause_descriptions"] == (1 if failure_kind == "no_pause" else 0)
    assert result.cues[0].playback_start == 1.0 and result.cues[0].tts_duration == 2.5
    assert result.cues[1].skip_reason
    assert Path(result.modified_audio_path).is_file()


@pytest.mark.parametrize("items", [[], [{"description": ""}], [{"description": "  "}, {"description": None}]])
def test_empty_narrative_is_not_tts_failure(voice, items):
    result = generate(voice, items)
    assert result.total_descriptions == result.inserted_descriptions == result.skipped_descriptions == 0
    assert result.diagnostics["outcome"] == "no_descriptions"
    assert result.diagnostics["technical_failures"] == 0
    voice.tts.save_to_file.assert_not_called()
    voice.scheduler.find_slot.assert_not_called()


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_audio_duration_is_technical_failure(voice, duration):
    voice.mixer.get_duration.return_value = duration
    with pytest.raises(voice.models.VoiceGenerationError) as raised:
        generate(voice)
    assert raised.value.diagnostics["audio_validation_failures"] == 1
    assert raised.value.diagnostics["tts_generation_failures"] == 0
    assert isinstance(raised.value.__cause__, ValueError)
    voice.scheduler.find_slot.assert_not_called()
    voice.mixer.mix.assert_not_called()


def test_unreadable_audio_is_distinct_from_no_pause(voice):
    original = ValueError("unreadable WAV")
    voice.mixer.get_duration.side_effect = original
    with pytest.raises(voice.models.VoiceGenerationError) as raised:
        generate(voice)
    assert raised.value.__cause__ is original
    assert raised.value.diagnostics["audio_validation_failures"] == 1
    assert raised.value.diagnostics["no_pause_descriptions"] == 0


def test_failed_tts_plus_no_pause_with_zero_insertions_fails(voice):
    synthesize = voice.tts.save_to_file.side_effect
    voice.tts.save_to_file.side_effect = [ConnectionError("provider down"), synthesize(text="x", output_path=str(voice.engine.cues_dir / "second.wav"))]
    voice.scheduler.find_slot.side_effect = None
    voice.scheduler.find_slot.return_value = None
    with pytest.raises(voice.models.VoiceGenerationError) as raised:
        generate(voice, timeline(2))
    assert raised.value.diagnostics["tts_generation_failures"] == 1
    assert raised.value.diagnostics["no_pause_descriptions"] == 1


def test_scheduling_exception_is_not_provider_failure(voice):
    voice.scheduler.find_slot.side_effect = RuntimeError("scheduler error")
    with pytest.raises(voice.models.VoiceGenerationError) as raised:
        generate(voice)
    assert raised.value.diagnostics["scheduling_failures"] == 1
    assert raised.value.diagnostics["tts_generation_failures"] == 0


def test_mixing_exception_propagates_with_original_cause(voice):
    original = OSError(r"C:\private\mix.wav SIMULATED_SECRET")
    voice.mixer.mix.side_effect = original
    with pytest.raises(voice.models.VoiceGenerationError) as raised:
        generate(voice)
    assert raised.value.__cause__ is original
    assert raised.value.diagnostics["outcome"] == "mixing_failed"
    assert "SIMULATED_SECRET" not in str(raised.value)
    assert not (voice.engine.manifest_dir / "voice_engine_manifest.json").exists()


def test_diagnostics_are_optional_for_legacy_constructors(voice):
    result = voice.models.VoiceEngineResult("original", "mixed", "manifest", 0, 0, 0, [])
    assert "diagnostics" not in result.to_dict()


@pytest.mark.parametrize("scenario,expected_state", [
    ("success", "completed"), ("partial", "completed"), ("no_pause", "completed"),
    ("empty", "completed"), ("tts_failed", "failed"), ("invalid_audio", "failed"),
    ("mix_failed", "failed"), ("tts_failed_and_no_pause", "failed"),
])
def test_voice_outcome_reaches_job_and_existing_http_contract(voice, api_modules, scenario, expected_state, caplog):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    api = api_modules
    voice.engine = voice.build(api.settings.generations_dir / "testjob" / "voice_engine")
    if scenario == "tts_failed":
        voice.tts.save_to_file.side_effect = ConnectionError("SIMULATED_SECRET C:/private/token")
    elif scenario == "invalid_audio":
        voice.mixer.get_duration.return_value = 0
    elif scenario == "mix_failed":
        voice.mixer.mix.side_effect = OSError("SIMULATED_SECRET C:/private/mix.wav")
    elif scenario == "no_pause":
        voice.scheduler.find_slot.side_effect = None
        voice.scheduler.find_slot.return_value = None
    elif scenario == "partial":
        good = str(voice.engine.cues_dir / "valid.wav")
        write_wav(Path(good), 2.5)
        voice.tts.save_to_file.side_effect = [ConnectionError("SIMULATED_SECRET"), good]
    elif scenario == "tts_failed_and_no_pause":
        good = write_wav(voice.engine.cues_dir / "valid.wav", 2.5)
        voice.tts.save_to_file.side_effect = [ConnectionError("provider failure"), good]
        voice.scheduler.find_slot.side_effect = None
        voice.scheduler.find_slot.return_value = None

    def pipeline(**kwargs):
        items = [] if scenario == "empty" else timeline(2 if scenario in ("partial", "tts_failed_and_no_pause") else 1)
        # Match the real orchestration: an empty timeline does not call VoiceEngine.
        result = generate(voice, items) if items else None
        return {"narrative": {"timeline": items}, "voice_engine": result.to_dict() if result else None,
                "artifacts": {"audio_description": result.modified_audio_path,
                              "voice_manifest": result.manifest_path} if result else {}}

    caplog.set_level("INFO")
    api.pipeline.process_video_verified.side_effect = pipeline
    api.service.run_generation_job("testjob", "unused", {})
    job = api.store.read_job("testjob")
    assert job["status"] == expected_state and job["finished_at"]
    app = FastAPI()
    app.include_router(api.routes.router)
    with TestClient(app) as client:
        status = client.get("/api/v1/generations/testjob")
        result_response = client.get("/api/v1/generations/testjob/result")
        audio = client.get("/api/v1/generations/testjob/audio-description")
    assert status.json()["status"] == expected_state
    if expected_state == "failed":
        assert result_response.status_code == audio.status_code == 500
        assert not (api.settings.generations_dir / "testjob/result.json").exists()
        assert "SIMULATED_SECRET" not in status.text + result_response.text + audio.text
        assert "C:/private" not in status.text + result_response.text + audio.text
    else:
        assert result_response.status_code == 200
        if scenario == "empty":
            assert audio.status_code == 404
            assert result_response.json()["result"]["voice_engine"] is None
            assert "voice_outcome=no_descriptions" in caplog.text
        else:
            assert audio.status_code == 200 and audio.content == Path(voice.original).read_bytes()
            assert Path(job["result_path"]).is_file()
            metadata = result_response.json()["result"]["voice_engine"]
            assert metadata["diagnostics"]["outcome"] == {
                "success": "success", "partial": "partial_success", "no_pause": "no_available_pause",
            }[scenario]


def test_job_persistence_failure_keeps_voice_error_and_provider_cause(voice, api_modules, monkeypatch, caplog):
    api = api_modules
    original = ConnectionError("provider root cause")
    voice.tts.save_to_file.side_effect = original
    api.pipeline.process_video_verified.side_effect = lambda **kwargs: generate(voice)
    real_update = api.store.update_job

    def update(job_id, **changes):
        if changes["status"] == "failed":
            raise PermissionError("secondary persistence error")
        return real_update(job_id, **changes)

    monkeypatch.setattr(api.service, "update_job", update)
    with pytest.raises(voice.models.VoiceGenerationError) as raised:
        api.service.run_generation_job("testjob", "unused", {})
    assert raised.value.__cause__ is original
    assert "provider root cause" in caplog.text and "secondary persistence error" in caplog.text
    assert "failed_state_persistence_failed" in caplog.text
