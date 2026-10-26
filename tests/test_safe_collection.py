"""Collectable examples must not import inference modules or process media."""
import ast
import importlib.abc
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_examples_import_without_ai_or_pipeline():
    # Separate interpreter avoids modules cached by unrelated tests. The import
    # guard rejects any attempt to load AI/pipeline code while importing examples.
    code = r'''
import importlib.abc, importlib.util, pathlib, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ai', 'pipeline'}:
            raise AssertionError('Heavy import during collection: ' + fullname)
sys.meta_path.insert(0, Guard())
for path in sorted(pathlib.Path('tests').glob('*.py')):
    if path.name.startswith(('test_', 'run_')) and path.name != 'test_person_no_age.py':
        spec = importlib.util.spec_from_file_location('collection_' + path.stem, path)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_diagnostic_settings_do_not_create_directories(monkeypatch):
    spec = importlib.util.spec_from_file_location("backend_check", ROOT / "tools/check_backend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic must not create directories")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    settings = module.load_settings()
    assert len(settings.get_model_paths()) == 6


def test_demo_environment_names_are_consumed():
    source = "\n".join((ROOT / p).read_text(encoding="utf-8") for p in (
        "pipeline/orchestration/verified_pipeline.py", "ai/audio_description/tts_client.py"))
    strings = {node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    cmd = (ROOT / "start_demo.cmd").read_text(encoding="utf-8")
    import re
    names = re.findall(r'set "(SEE2SOUND_\w+)=', cmd)
    assert len(names) == 10 and set(names).issubset(strings)
    assert 'set "SEE2SOUND_TTS_RATE=-5%%"' in cmd
    assert "--host 127.0.0.1 --port 8000 --workers 1" in cmd


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CMD only")
def test_cmd_profile_preserves_literal_percent(tmp_path):
    # Execute the real prefix of the launcher, stopping before any Python/API call.
    (tmp_path / "API").mkdir()
    (tmp_path / "API/main.py").touch()
    (tmp_path / "pipeline/orchestration").mkdir(parents=True)
    (tmp_path / "pipeline/orchestration/verified_pipeline.py").touch()
    source = (ROOT / "start_demo.cmd").read_text(encoding="utf-8")
    prefix = source.split('set "DEMO_PYTHON=python"', 1)[0]
    probe = tmp_path / "profile_probe.cmd"
    probe.write_text(prefix + "set SEE2SOUND_\nexit /b 0\n", encoding="ascii")
    result = subprocess.run(["cmd.exe", "/d", "/c", str(probe)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    values = dict(line.split("=", 1) for line in result.stdout.splitlines()
                  if line.startswith("SEE2SOUND_"))
    assert values["SEE2SOUND_TTS_RATE"] == "-5%"
    assert values["SEE2SOUND_TTS_VOICE"] == "pt-BR-FranciscaNeural"
    assert values["SEE2SOUND_PERSON_INTERMEDIATE"] == "true"
    assert values["SEE2SOUND_OBJECT_APPLY_ADDITIONS"] == "false"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CMD only")
def test_demo_rejects_wrong_working_directory(tmp_path):
    result = subprocess.run(["cmd.exe", "/d", "/c", str(ROOT / "start_demo.cmd")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "ERRO: execute start_demo.cmd na raiz" in result.stdout
