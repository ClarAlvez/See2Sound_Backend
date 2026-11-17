"""Examina nomes/VoiceEngine e recupera pausas estendidas sem repetir as IAs.

Na raiz do backend:
  python -m tools.postprocess_v07 --result data/output/integration_v07/pipeline_result_v07.json
  python -m tools.postprocess_v07 --result data/output/integration_v07/pipeline_result_v07.json --apply
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

NO_SLOT = "Nenhum espaço de áudio suficientemente longo foi encontrado."


def summarize(result: dict) -> dict:
    corr = result.get("correlation") or {}
    identity = corr.get("audio_identity") or {}
    candidates = identity.get("name_candidates") or {}
    if not isinstance(candidates, dict):
        candidates = {}
    unassigned = identity.get("unassigned_name_mentions") or []
    mentions = sum(len(x.get("name_mentions") or []) for x in identity.get("segments") or [] if isinstance(x, dict))
    confirmed = identity.get("confirmed_names") or {}
    voice = result.get("voice_engine") or {}
    cues = voice.get("cues") or []
    reason_counts = Counter(str(c.get("skip_reason")) for c in cues if not c.get("inserted"))
    extended = sum(c.get("mode") == "extended_pause" for c in cues)
    return {
        "speech_segments": len(identity.get("segments") or []),
        "name_mentions": mentions,
        "candidate_entities": len(candidates),
        "candidate_names": {eid: [x.get("name") for x in items if isinstance(x, dict)]
                            for eid, items in candidates.items() if isinstance(items, list)},
        "unassigned_mentions": len(unassigned),
        "confirmed_names": confirmed,
        "speaker_links_status": (corr.get("speaker_links") or {}).get("status") if isinstance(corr.get("speaker_links"), dict) else None,
        "narrative_descriptions": len((result.get("narrative") or {}).get("timeline") or []),
        "voice_cues": len(cues),
        "voice_inserted": sum(bool(c.get("inserted")) for c in cues),
        "voice_skipped": sum(not c.get("inserted") for c in cues),
        "no_available_pause": sum(not c.get("inserted") and c.get("skip_reason") == NO_SLOT for c in cues),
        "skip_reasons": dict(reason_counts),
        "extended_pause_cues": extended,
        "video_with_pauses": (result.get("artifacts") or {}).get("video_with_pauses"),
        "playback": result.get("playback"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default="data/output/integration_v07/pipeline_result_v07.json")
    parser.add_argument("--video", help="Caminho original, se source_video_path estiver ausente")
    parser.add_argument("--apply", action="store_true", help="Executa pausas estendidas e atualiza JSON")
    args = parser.parse_args()
    path = Path(args.result).resolve()
    if not path.is_file():
        parser.error(f"Resultado não encontrado: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        parser.error("JSON deve conter objeto raiz")
    print("=== Diagnóstico antes ===")
    print(json.dumps(summarize(result), indent=2, ensure_ascii=False, default=str))
    if not args.apply:
        print("\nModo inspeção: nada foi alterado. Acrescente --apply para recuperar as pausas.")
        return
    if not result.get("voice_engine"):
        parser.error("Resultado sem voice_engine. Execute a pipeline com TTS habilitado.")
    original_video = args.video or result.get("source_video_path")
    if not original_video or not Path(original_video).is_file():
        parser.error("Vídeo original não encontrado; informe --video \"caminho.mp4\".")
    eligible = summarize(result)["no_available_pause"]
    if not eligible:
        print("Sem cues pendentes por falta de pausa; nenhum congelamento necessário.")
        return
    if shutil.which("ffmpeg") is None:
        parser.error("FFmpeg não está disponível no PATH. Execute 'where.exe ffmpeg' primeiro.")
    from ai.voice_engine.extended_playback import extend_video_when_needed
    did_extend = extend_video_when_needed(
        video_path=str(Path(original_video).resolve()),
        result=result, output_dir=path.parent)
    if not did_extend:
        print("O módulo de pausas não criou novos artefatos; confira se os cues possuem WAV e duração válidos.")
        return
    backup = path.with_name(path.stem + ".before_extended_playback.json")
    if not backup.exists():
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    print("\n=== Diagnóstico depois ===")
    print(json.dumps(summarize(result), indent=2, ensure_ascii=False, default=str))
    print("JSON atualizado:", path)
    print("Backup:", backup)


if __name__ == "__main__":
    main()
