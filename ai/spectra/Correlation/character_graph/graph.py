"""CharacterGraph v1.0: voice-first names + face clusters + verified AV edges.

Consumes CACHED Correlation output. Model inference is optional and isolated to
face embedding on existing person crops; diarization/ASD are NOT re-executed.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

from .voice import build_voice_identities
from .visual import build_visual_identities,tracklet_key
from .fusion import resolve_voice_visual_links


def _make_face_encoder(config, warnings):
    if not getattr(config,'character_face_enabled',True):
        warnings.append('SFace desativado por configuracao; mantendo rastros visuais separados.')
        return None
    root=Path('data/models/Face')
    detector=Path(getattr(config,'character_yunet_path',root/'face_detection_yunet_2023mar.onnx'))
    recognizer=Path(getattr(config,'character_sface_path',root/'face_recognition_sface_2021dec.onnx'))
    if not detector.is_file() or not recognizer.is_file():
        warnings.append('YuNet/SFace ONNX indisponiveis: face Re-ID nao executado. '
                        'Execute python -m tools.install_face_models_v10.')
        return None
    try:
        from .face_encoder import SFaceEncoder
        return SFaceEncoder(detector,recognizer)
    except Exception as exc:
        warnings.append(f'YuNet/SFace nao inicializado: {type(exc).__name__}: {exc}')
        return None


def build_character_graph(audio_identity, frames, entities, active_speaker,
                          *, diarization=None, config=None,
                          manual_verified_map=None, crop_root=None,
                          face_encoder=None, disable_face_models=False, entity_remap=None):
    warnings=[]
    if face_encoder is None and not disable_face_models:
        face_encoder=_make_face_encoder(config,warnings)
    elif face_encoder is None:
        warnings.append('Face Re-ID desativado neste replay.')
    segments=(audio_identity or {}).get('segments') or []
    voice_nodes=build_voice_identities(segments,diarization=diarization)
    anchor_times=[mention.get('timestamp') for v in voice_nodes.values()
                  for mention in v.get('name_mentions',[])
                  if mention.get('kind')=='self_introduction']
    visual_nodes,entity_to_visual,face_stats,tracklet_to_visual,entity_timeline=build_visual_identities(
        frames,entities,encoder=face_encoder,crop_root=crop_root,
        max_entities=int(getattr(config,'character_max_face_entities',120)),
        max_crops_per_entity=int(getattr(config,'character_max_face_crops',3)),
        min_similarity=float(getattr(config,'character_min_face_similarity',0.55)),
        strong_similarity=float(getattr(config,'character_strong_face_similarity',0.65)),
        attention_times=anchor_times)
    # Remap manual legacy IDs through the reconciler, never as guesses.
    reviewed_map={}
    aliases=entity_remap or {}
    def canonical(eid):
        seen=set()
        while eid in aliases and eid not in seen:
            seen.add(eid)
            eid=aliases[eid]
        return eid
    for sid,val in (manual_verified_map or {}).items():
        if isinstance(val,str):
            reviewed_map[sid]=canonical(val) if not val.startswith('visual_') else val
        elif isinstance(val,dict):
            cleaned=dict(val)
            if cleaned.get('entity_id'):
                cleaned['entity_id']=canonical(cleaned['entity_id'])
            reviewed_map[sid]=cleaned
        else:
            reviewed_map[sid]=val
    speaker_links=resolve_voice_visual_links(
        active_speaker,voice_nodes,visual_nodes,entity_to_visual,
        min_turns=int(getattr(config,'character_min_asd_turns',3)),
        manual_verified_map=reviewed_map,entity_timeline=entity_timeline)
    confirmed_visual={}
    name_collisions=defaultdict(set)
    characters={}
    for sid,voice in voice_nodes.items():
        link=speaker_links['links'].get(sid) or {}
        # Unconfirmed name follows only the voice, not the visual candidate.
        visual=link.get('visual_id') if link.get('verified') else None
        voice['visual_id']=visual
        label=voice['name_claim']
        if not visual:
            voice['visual_name_status']='unlinked_voice'
            continue
        voice['visual_name_status']='verified_av' if label else 'verified_av_unnamed'
        if label:
            name_collisions[visual].add(label)
            characters.setdefault(visual,[]).append((sid,label))
    for vid,pairs in characters.items():
        names={name.casefold() for _,name in pairs}
        if len(names)==1:
            name=pairs[0][1]
            confirmed_visual[vid]=name
            visual_nodes[vid]['resolved_name']=name
        else:
            warnings.append(f'Nomes autodeclarados conflitantes na identidade {vid}.')
    # A single self-declared name on unrelated visual clusters is ambiguous.
    # Without diarization speaker merge, do not claim both are that person.
    visual_by_name=defaultdict(set)
    for visual_id,label in confirmed_visual.items():
        visual_by_name[label.casefold()].add(visual_id)
    for name,vids in visual_by_name.items():
        if len(vids)>1:
            for vid in vids:
                confirmed_visual.pop(vid,None)
                visual_nodes[vid]['resolved_name']=None
            warnings.append('Nome autodeclarado em multiplas identidades visuais: '+name)
    # Character nodes also exist for named voices with no visible counterpart.
    # Thus "Sou Eleanor" is useful without inventing which face belongs to her.
    characters_by_id={}
    claimed_visual=set()
    for sid,voice in sorted(voice_nodes.items()):
        vid=voice.get('visual_id')
        if not vid and not voice.get('name_claim'):
            continue
        cid='character_voice_'+sid
        status=('verified_voice_visual' if vid and vid in confirmed_visual and voice.get('name_claim')
                else 'voice_visual_unnamed' if vid else 'voice_only')
        characters_by_id[cid]={
            'character_id':cid, 'name':voice.get('name_claim'),
            'name_status':('confirmed_visual' if status=='verified_voice_visual' else
                           'self_declared_voice_only' if voice.get('name_claim') else 'unknown'),
            'voice_id':sid,'visual_id':vid,
            'entity_ids':visual_nodes[vid]['entity_ids'] if vid else [],
            'status':status,
        }
        voice['character_id']=cid
        if vid:claimed_visual.add(vid)
    for vid,visual in visual_nodes.items():
        if vid in claimed_visual:
            continue
        cid='character_visual_'+vid
        characters_by_id[cid]={
            'character_id':cid,'name':None,'name_status':'unknown',
            'voice_id':None,'visual_id':vid,'entity_ids':visual['entity_ids'],
            'status':'visual_only',
        }
    visual_to_character={row['visual_id']:cid for cid,row in characters_by_id.items()
                         if row['visual_id']}
    # Do not reuse one voice node for multiple visual identities; fusion forbids
    # this. Do not reject the same named character seen in several source IDs
    # if those IDs are inside one face cluster.
    # Old per-person alias API only receives globally unambiguous identities.
    # If old ResNet conflated two people under one ID, NEVER rename all frames.
    confirmed_entity={eid:name for vid,name in confirmed_visual.items()
                      for eid in visual_nodes[vid]['entity_ids']
                      if entity_to_visual.get(eid)==vid}
    updated_audio=dict(audio_identity or {})
    updated_audio['identity_version']='character_graph_v1'
    updated_audio['voice_identities']=voice_nodes
    updated_audio['visual_name_candidates']={}  # no speculative person names
    updated_audio['name_candidates']={eid:[{'name':name,'status':'confirmed',
                                           'evidence_types':['voice_self_claim','verified_av_link'],
                                           'verified_evidence_count':1}]
                                      for eid,name in confirmed_entity.items()}
    updated_audio['confirmed_names']=confirmed_entity
    # Replace previous per-mention heuristic entity assignment. Names spoken by
    # another person remain references, never identities of that speaker.
    unassigned=[]
    for seg in updated_audio.get('segments') or []:
        sid=str(seg.get('speaker_id') or '')
        node=voice_nodes.get(sid,{})
        v= node.get('visual_id')
        seg['speaker_visual_id']=v
        seg['speaker_character_id']=node.get('character_id')
        seg['speaker_entity_id']=speaker_links['links'].get(sid,{}).get('entity_id') if v else None
        seg['speaker_grounding_verified']=bool(v)
        seg['speaker_name_claim']=node.get('name_claim')
        for mention in seg.get('name_mentions') or []:
            candidate_name=node.get('name_claim')
            self_verified=(mention.get('kind')=='self_introduction' and
                           candidate_name is not None and
                           candidate_name.casefold()==str(mention.get('name','')).casefold() and
                           v in confirmed_visual)
            mention.update({'entity_id':seg['speaker_entity_id'] if self_verified else None,
                            'verified':bool(self_verified),
                            'evidence_type':('voice_self_claim_plus_verified_av' if self_verified
                                             else 'voice_only_or_reference'),
                            'heuristic_score':1.0 if self_verified else 0.0,
                            'voice_id':sid})
            if not self_verified:
                unassigned.append(mention)
    updated_audio['unassigned_name_mentions']=unassigned
    for scene_rows in (updated_audio.get('dialogue_by_scene') or {}).values():
        for row in scene_rows:
            if not isinstance(row,dict):
                continue
            original=next((seg for seg in updated_audio.get('segments') or []
                           if seg.get('segment_id')==row.get('segment_id')),None)
            if original:
                row['speaker_entity_id']=original['speaker_entity_id']
                row['speaker_character_id']=original.get('speaker_character_id')
                row['name_mentions']=original.get('name_mentions',[])
    for frame in frames or []:
        for p in frame.get('people') or []:
            eid=str(p.get('entity_id') or '')
            vid=tracklet_to_visual.get(tracklet_key(frame,p))
            p['visual_id']=vid
            p['character_id']=visual_to_character.get(vid)
            p['resolved_name']=confirmed_visual.get(vid)
            # Existing anonymous display_name is not overwritten unless named.
            if vid in confirmed_visual:
                p['display_name']=confirmed_visual[vid]
    graph={
        'version':'1.0.1',
        'policy':'voice_first_face_reid_verified_av',
        'voice_identities':voice_nodes,
        'characters':characters_by_id,
        'visual_identities':visual_nodes,
        'entity_to_visual':entity_to_visual,
        'tracklet_to_visual':tracklet_to_visual,
        'voice_visual_links':speaker_links['links'],
        'confirmed_names_by_visual':confirmed_visual,
        'confirmed_names_by_entity':confirmed_entity,
        'diagnostics':{
            'voices':len(voice_nodes), 'voices_with_self_claims':sum(bool(v['name_claim']) for v in voice_nodes.values()),
            'visual_clusters':len(visual_nodes),'confirmed_characters':len(confirmed_visual),
            'unambiguous_legacy_name_aliases':len(confirmed_entity),
            'face':face_stats,'links':speaker_links['status_counts'],
            'warnings':warnings,
        },
    }
    return graph,updated_audio,speaker_links
