"""Patch safety without requiring the full backend runtime."""
from __future__ import annotations
from pathlib import Path
import importlib.util

FILE=Path(__file__).resolve().parents[1]/'patch_pipeline.py'
spec=importlib.util.spec_from_file_location('patch_pipeline_v06',FILE)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_patch_rejects_unexpected_pipeline():
    try:
        module.patch_text('def process_video(): return {}')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Nao pode editar codigo inesperado')


def test_patch_inserts_compatible_opt_in_flags():
    original=("def process_video(\n    run_spectra: bool = True,\n    run_narrative: bool = True,\n):\n"
              "    narrative_timeline: List[\n        Dict[str, Any]\n    ] = []\n"
              "    return build_processing_result(\n        a=1,\n    )\n")
    changed,yes=module.patch_text(original)
    assert yes and 'correlation_run_diarization' in changed
    assert 'correlation_run_active_speaker' in changed
    assert "result['correlation'] = correlation_result" in changed
    second,no=module.patch_text(changed)
    assert not no and second==changed
