"""Reavalia nomes/vinculos com JSON salvo, sem reexecutar modelos.

Exemplos:
  python -m tools.replay_identity_v08 --result data/output/integration_v07/pipeline_result_v07.json
  python -m tools.replay_identity_v08 --result ... --verify-map reviewed.json --narrative-preview
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def inspect_identity(result):
    c = result.get('correlation') or {}
    a = c.get('audio_identity') or {}
    l = (c.get('speaker_links') or {}).get('links') or {}
    mentions = [m for seg in a.get('segments') or [] for m in seg.get('name_mentions') or []]
    return {
        'segments': len(a.get('segments') or []),
        'name_mentions': len(mentions),
        'mentions_by_type': dict(Counter(x.get('kind') for x in mentions)),
        'unassigned_mentions': len(a.get('unassigned_name_mentions') or []),
        'candidate_names': {eid:[x.get('name') for x in names]
                            for eid,names in (a.get('name_candidates') or {}).items()},
        'confirmed_names': a.get('confirmed_names') or {},
        'speaker_links_status': dict(Counter(v.get('status') for v in l.values())),
        'linked_speakers': {sid:link.get('entity_id') for sid,link in l.items()
                            if link.get('entity_id')},
    }


def replay(result, *, verified_map=None, auto_confirm=False):
    from ai.spectra.Correlation.config import CorrelationConfig
    from ai.spectra.Correlation.audio.speaker_linking import SpeakerLinkMemory
    from ai.spectra.Correlation.speech_identity import SpeechIdentityResolver
    corr = result.get('correlation') or {}
    if not corr.get('frames') or not corr.get('entities'):
        raise ValueError('JSON nao possui correlation.frames e correlation.entities; precisa de resultado completo v0.6/v0.7.')
    if not (corr.get('audio_identity') or {}).get('segments'):
        raise ValueError('JSON nao possui correlation.audio_identity.segments para replay.')
    old = corr['audio_identity']
    raw_scores = (corr.get('active_speaker') or {}).get('turn_scores') or []
    config = CorrelationConfig()
    links = SpeakerLinkMemory(
        min_turns=3 if auto_confirm else config.speaker_min_confirming_turns,
        min_support_fraction=.85 if auto_confirm else config.speaker_min_support_fraction,
        min_face_coverage=.65 if auto_confirm else config.speaker_min_face_coverage,
        allow_automatic_name_confirmation=auto_confirm,
    ).resolve(raw_scores, valid_entity_ids=corr['entities'])
    speaker_map = dict(links['speaker_entity_map'])
    # Somente mapa conferido por um humano pode usar verified=True.
    if verified_map:
        for speaker, value in verified_map.items():
            eid = value if isinstance(value,str) else value.get('entity_id')
            if eid not in corr['entities']:
                raise ValueError(f'{speaker}: pessoa inexistente: {eid}')
            speaker_map[str(speaker)] = {'entity_id':eid,'verified':True,
                                         'method':'human_review_v08'}
    speech_analysis = dict((result.get('audio') or {}).get('speech_analysis') or {})
    # O JSON original do Whisper nao retém necessariamente os speaker_id; a
    # saida alinhada do audio_identity e a fonte correta para replay.
    speech_analysis['speech_segments'] = [
        {key: seg.get(key) for key in ('segment_id','start','end','text','speaker_id','words')}
        for seg in old['segments']
    ]
    identity = SpeechIdentityResolver(config).resolve(
        speech_analysis, corr['frames'], corr['entities'],
        pipeline_payload=result, speaker_entity_map=speaker_map,
    )
    corr['audio_identity'] = identity
    corr['speaker_links'] = links
    corr['speaker_links']['reviewed_overrides'] = sorted(verified_map or {})
    names = identity['confirmed_names']
    for eid, profile in (corr.get('entity_profiles') or {}).items():
        profile['name_candidates'] = identity['name_candidates'].get(eid, [])
        profile['resolved_name'] = names.get(eid)
        profile['name_status'] = ('confirmed' if eid in names else
                                  'candidate' if profile['name_candidates'] else 'unknown')
        if eid in names:
            profile['display_name'] = names[eid]
    for frame in corr['frames']:
        for person in frame.get('people') or []:
            eid = person.get('entity_id')
            person['resolved_name'] = names.get(eid)
            if eid in names:
                person['display_name'] = names[eid]
    # Os textos antigos do Narrative, Voice Engine e MP4 estendido nao foram
    # regenerados: jamais declarar que eles mudaram com um replay de identidade.
    result.setdefault('identity_v08_diagnostics', {})['replayed_from_cached_data'] = True
    result['identity_v08_diagnostics']['narrative_and_video_unchanged'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True)
    parser.add_argument('--output',help='Novo JSON, sem sobrescrever o original')
    parser.add_argument('--verify-map',help='JSON speaker_id -> entity_id, SOMENTE apos revisao visual')
    parser.add_argument('--auto-confirm',action='store_true',help='Experimental: confirmar apresentacoes so com ASD forte (3+ turnos independentes)')
    parser.add_argument('--narrative-preview',action='store_true',help='Previa de texto sem Qwen nem TTS')
    parser.add_argument('--focus-narrative',action='store_true',help='Filtrar a previa por relevancia, sem TTS')
    parser.add_argument('--max-descriptions-per-minute',type=int,default=4)
    args = parser.parse_args()
    source = Path(args.result).resolve()
    if not source.is_file():
        parser.error(f'Resultado inexistente: {source}')
    result = json.loads(source.read_text(encoding='utf8'))
    overrides = None
    if args.verify_map:
        overrides = json.loads(Path(args.verify_map).read_text(encoding='utf8'))
        if not isinstance(overrides,dict):
            parser.error('--verify-map exige JSON objeto {"SPEAKER_02": "person_0009"}')
    print('=== ANTES ===')
    print(json.dumps(inspect_identity(result),ensure_ascii=False,indent=2))
    result = replay(result,verified_map=overrides,auto_confirm=args.auto_confirm)
    print('=== DEPOIS ===')
    print(json.dumps(inspect_identity(result),ensure_ascii=False,indent=2))
    output = Path(args.output).resolve() if args.output else source.with_name(source.stem+'_identity_v08.json')
    if output == source:
        parser.error('Nunca sobrescrever o JSON original com este replay.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    print('Resultado novo:',output)
    if args.narrative_preview:
        from ai.narrative.correlation_narrative import CorrelationNarrativeGenerator
        gen = CorrelationNarrativeGenerator(use_llm=False)
        timeline = gen.generate_timeline_from_dicts((result.get('spectra') or {}).get('outputs') or [],
                                                     correlation_result=result['correlation'])
        if args.focus_narrative:
            from ai.narrative.focus_filter_v08 import select_focused_descriptions
            timeline = select_focused_descriptions(
                timeline, correlation_result=result['correlation'],
                max_per_minute=args.max_descriptions_per_minute)
        path = output.with_name(output.stem+'_narrative_preview.json')
        path.write_text(json.dumps(timeline,ensure_ascii=False,indent=2),encoding='utf8')
        print('Narrative sem LLM/TTS:',path)
        for row in timeline[:20]:
            print(f"{row.get('start_time')}: {row.get('description')}")


if __name__=='__main__':
    main()
