"""Reprocessa CharacterGraph em JSON Correlation salvo, sem IA de video/audio.

Por padrao nao sobrescreve o JSON original, nem o Narrative ou MP4 existentes.
Pode executar YuNet+SFace nos crops ja salvos se pesos estiverem instalados.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def replay(data, *, verified_map=None, disable_face=False, crop_root=None):
    from ai.spectra.Correlation.character_graph import build_character_graph
    from ai.spectra.Correlation.config import CorrelationConfig
    corr=data.get('correlation',data)
    if not isinstance(corr,dict):
        raise ValueError('Resultado invalido.')
    for required in ('frames','entities','audio_identity','active_speaker'):
        if required not in corr:
            raise ValueError('JSON sem campo '+required+'; requer saida Correlation v0.6 ou v0.8.')
    graph,identity,links=build_character_graph(
        corr['audio_identity'],corr['frames'],corr['entities'],corr['active_speaker'],
        diarization=corr.get('audio_diarization'),config=CorrelationConfig(),
        manual_verified_map=verified_map,crop_root=crop_root,
        entity_remap=(corr.get('reconciliation') or {}).get('entity_remap'),
        disable_face_models=disable_face)
    corr['character_graph']=graph
    corr['audio_identity']=identity
    corr['legacy_speaker_links_v08']=corr.get('speaker_links')
    corr['speaker_links']=links
    corr['version']='1.0.0'
    names=identity['confirmed_names']
    for eid,profile in (corr.get('entity_profiles') or {}).items():
        profile['visual_id']=graph['entity_to_visual'].get(eid)
        profile['resolved_name']=names.get(eid)
        profile['name_candidates']=identity['name_candidates'].get(eid,[])
        profile['name_status']='confirmed' if eid in names else 'unknown'
        if eid in names:
            profile['display_name']=names[eid]
    # Existing narrative text and video remain unchanged.
    data['character_graph_v10_replay']={
        'from_cached_correlation':True,
        'spectra_whisper_pyannote_asd_not_rerun':True,
        'narrative_tts_video_not_regenerated':True,
    }
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--correlation',required=True,help='correlation_v06.json ou pipeline_result_v07.json')
    p.add_argument('--output',help='JSON de saida (padrao *_character_v10.json)')
    p.add_argument('--verified-map',help='Mapa humano: {"SPEAKER_02": "person_XXXX"}')
    p.add_argument('--skip-face',action='store_true',help='Nao usar SFace mesmo com modelos locais')
    p.add_argument('--crop-root',help='Diretorio base alternativo para person_crops')
    args=p.parse_args()
    source=Path(args.correlation).resolve()
    if not source.is_file():p.error('JSON nao encontrado: '+str(source))
    output=Path(args.output).resolve() if args.output else source.with_name(source.stem+'_character_v10.json')
    if output==source:p.error('Nao sobrescrever o JSON original.')
    data=json.loads(source.read_text(encoding='utf8'))
    reviewed=None
    if args.verified_map:
        reviewed=json.loads(Path(args.verified_map).read_text(encoding='utf8'))
        if not isinstance(reviewed,dict):p.error('Mapa de verificacao deve ser dict.')
    data=replay(data,verified_map=reviewed,disable_face=args.skip_face,crop_root=args.crop_root)
    corr=data.get('correlation',data)
    graph=corr['character_graph']
    print('\n=== CHARACTERGRAPH V1.0 ===')
    print(json.dumps(graph['diagnostics'],indent=2,ensure_ascii=False))
    print('\n=== NOMES DAS VOZES ===')
    for sid,voice in graph['voice_identities'].items():
        if voice.get('name_claim'):
            link=graph['voice_visual_links'].get(sid,{})
            print(sid,'->',voice['name_claim'],'| voz:',voice['name_status'],
                  '| visual:', link.get('visual_id') or link.get('candidate_visual_id'),
                  '| status:',link.get('status'))
    print('\n=== NOMES VISUAIS CONFIRMADOS ===')
    print(json.dumps(graph['confirmed_names_by_visual'],ensure_ascii=False,indent=2))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf8')
    print('\nArquivo gerado:',output)
    print('Narrative e video originais NAO foram alterados.')

if __name__=='__main__':main()
