"""Teste completo: Whisper → 5 Spectra → Correlation 0.6 → Narrative 0.7 → Voice.

Execute na raiz do See2Sound_Backend:
    python tools/run_pipeline_v07.py --video "data/test/video.mp4" --dry-run
    python tools/run_pipeline_v07.py --video "data/test/video.mp4"
"""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path

MODEL_FILENAMES = {
    'scene': 'scene_net_best.pt',
    'person': 'person_net_best.pt',
    'object': 'object_net_best.pt',
    'atmosphere': 'atmosphere_net_best.pt',
    'action': 'action_net_best.pt',
}


def find_model(kind: str, explicit: str | None, directory: Path) -> Path | None:
    if explicit:
        return Path(explicit)
    if not directory.is_dir():
        return None
    matches = sorted(directory.rglob(MODEL_FILENAMES[kind]))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f'Mais de um checkpoint para {kind}: {matches}. Passe --{kind}-model.')
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output-dir', default='data/output/integration_v07')
    parser.add_argument('--models-dir', default='data/models')
    for kind in MODEL_FILENAMES:
        parser.add_argument(f'--{kind}-model', default=None)
    parser.add_argument('--narrative-model', default=None)
    parser.add_argument('--max-frames-per-scene', type=int, default=24)
    parser.add_argument('--skip-asd', action='store_true', help='Desabilita somente Light-ASD')
    parser.add_argument('--skip-diarization', action='store_true', help='Desabilita somente pyannote')
    # SEE2SOUND_CORRELATION_V08_TEST
    parser.add_argument('--auto-confirm-names', action='store_true',
                        help='Experimental: confirmar apresentacoes apenas com vinculos ASD muito fortes')
    parser.add_argument('--focus-narrative', action='store_true',
                        help='Selecionar descricoes relevantes por personagem, antes do TTS')
    parser.add_argument('--max-descriptions-per-minute', type=int, default=4)
    parser.add_argument('--skip-tts', action='store_true', help='Testar até Narrative, sem Voice Engine')
    parser.add_argument('--progress-interval', type=float, default=10.0,
                        help='Intervalo entre atualizacoes no terminal (segundos, >= 1)')
    parser.add_argument('--dry-run', action='store_true', help='Verifica arquivos sem executar modelos')
    args = parser.parse_args()

    video = Path(args.video)
    if not video.is_file():
        parser.error(f'Vídeo inexistente: {video}')
    orchestrator = Path('pipeline/orchestration/process_video.py')
    if not orchestrator.is_file() or 'SPECTRA_NARRATIVE_V07' not in orchestrator.read_text(encoding='utf-8'):
        parser.error('Narrative v0.7 não instalado no orquestrador; execute install_overlay.py primeiro.')
    model_root = Path(args.models_dir)
    models = {kind: find_model(kind, getattr(args, f'{kind}_model'), model_root)
              for kind in MODEL_FILENAMES}
    qwen = Path(args.narrative_model) if args.narrative_model else Path(
        'data/models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf')
    if not qwen.is_file() and not args.narrative_model:
        choices = sorted(model_root.rglob('*.gguf')) if model_root.is_dir() else []
        if len(choices) == 1:
            qwen = choices[0]
    missing = [f'{kind}: {path if path else "não encontrado"}'
               for kind, path in models.items() if path is None or not path.is_file()]
    if not qwen.is_file():
        missing.append(f'Qwen: {qwen}')
    if not args.skip_asd:
        root = Path('third_party/Light-ASD')
        if not (root / 'weight/pretrain_AVA_CVPR.model').is_file():
            missing.append(f'Light-ASD: {root / "weight/pretrain_AVA_CVPR.model"}')
    print('See2Sound — teste de integração v0.7')
    print('Vídeo:', video)
    print('Saída:', args.output_dir)
    for kind, path in models.items():
        print(f'  {kind}: {path}')
    print('  narrative:', qwen)
    print('  diarização:', not args.skip_diarization)
    print('  ASD:', not args.skip_asd)
    print('  Voice/TTS:', not args.skip_tts)
    if missing:
        raise SystemExit('Pré-requisitos ausentes:\n' + '\n'.join(' - ' + x for x in missing))
    if args.dry_run:
        print('Pré-requisitos de arquivos OK. Nenhum modelo executado.')
        return

    # O import é tardio: --dry-run não carrega Torch/Qwen/Whisper.
    from pipeline.orchestration.process_video import process_video
    signature = inspect.signature(process_video)
    required = {'run_correlation', 'correlation_run_diarization', 'correlation_run_active_speaker'}
    absent = sorted(required.difference(signature.parameters))
    if absent:
        raise RuntimeError('Bridge da v0.6 ainda não instalada: ' + ', '.join(absent))
    # SEE2SOUND_PROGRESS_V1
    from tools.pipeline_progress_v1 import ProgressMonitor
    import shutil as _progress_shutil
    if args.progress_interval < 1:
        parser.error("--progress-interval deve ser >= 1 segundo")
    if not args.skip_tts and _progress_shutil.which("ffmpeg") is None:
        parser.error("FFmpeg ausente no PATH. Verifique com where.exe ffmpeg antes da inferencia completa.")
    with ProgressMonitor(Path(args.output_dir), interval_seconds=args.progress_interval,
                         run_tts=not args.skip_tts) as monitor:
        result = process_video(
            video_path=str(video),
            output_base_dir=args.output_dir,
            scene_model_path=str(models['scene']),
            person_model_path=str(models['person']),
            object_model_path=str(models['object']),
            atmosphere_model_path=str(models['atmosphere']),
            action_model_path=str(models['action']),
            strict_model_loading=True,
            use_person_cropper=True,
            use_object_cropper=True,
            use_action_model=True,
            use_action_person_cropper=True,
            max_temporal_frames_per_scene=args.max_frames_per_scene,
            run_spectra=True,
            run_correlation=True,
            correlation_run_diarization=not args.skip_diarization,
            correlation_run_active_speaker=not args.skip_asd,
            correlation_auto_confirm_names=args.auto_confirm_names,
            narrative_focus_mode=args.focus_narrative,
            narrative_max_per_minute=args.max_descriptions_per_minute,
            run_narrative=True,
            narrative_model_path=str(qwen),
            run_tts=not args.skip_tts,
            progress_callback=monitor.on_stage,
        )
        # Checkpoint anterior a FFmpeg para recuperar dados se renderizacao falhar.
        _checkpoint = Path(args.output_dir) / "pipeline_result_v07_pre_render.json"
        _checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _checkpoint.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        # SEE2SOUND_V07_EXTENDED_PLAYBACK_HOTFIX
        # Mesmo pós-processamento usado no fluxo API/services/generation_service.py:
        # salva vídeo e WAV estendidos ANTES de persistir pipeline_result_v07.json.
        if not args.skip_tts:
            monitor.on_stage("extended_playback")
            from ai.voice_engine.extended_playback import extend_video_when_needed
            extended = extend_video_when_needed(
                video_path=str(video.resolve()),
                result=result,
                output_dir=Path(args.output_dir),
            )
            print('Pausas de vídeo estendidas:', 'sim' if extended else 'não necessárias')

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / 'pipeline_result_v07.json'
        monitor.on_stage("saving_result")
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        correlation = result.get('correlation') or {}
        narrative = (result.get('narrative') or {}).get('timeline') or []
        print('\n=== Resumo da integração ===')
        print('Pessoas:', correlation.get('entity_count'))
        print('Objetos:', correlation.get('object_count'))
        print('Cenas:', len(correlation.get('scenes') or {}))
        print('Nomes confirmados:', (correlation.get('audio_identity') or {}).get('confirmed_names'))
        print('Descrição gerada:', len(narrative))
        for item in narrative[:15]:
            print(f"[{item.get('start_time')}–{item.get('end_time')}] {item.get('description')}")
        print('Voice Engine:', 'habilitada' if not args.skip_tts else 'desativada')
        print('Resultado completo:', result_path)
        print('Correlation:', (result.get('artifacts') or {}).get('correlation_v06_json'))


if __name__ == '__main__':
    main()
