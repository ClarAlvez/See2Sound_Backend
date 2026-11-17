"""Conservative temporal voice ↔ visual identity matching from Light-ASD.

ASD logits are NOT probabilities.  Require repeated independent lip-sync
observations and competing-speaker checks. Self-declared names stay on voice
nodes if no robust visual link is available; never equate the visible listener
with the speaker.
"""
from __future__ import annotations
from collections import defaultdict
from math import isfinite


def resolve_voice_visual_links(active_speaker, voice_nodes, visual_nodes, entity_to_visual,
                               *, min_turns=3, min_support_fraction=0.75,
                               min_face_coverage=0.55, min_face_frames=12,
                               min_raw_score=0., min_runner_margin=0.25,
                               manual_verified_map=None,entity_timeline=None,max_time_gap=2.25):
    raw=(active_speaker or {}).get('turn_scores') or []
    rejected=defaultdict(int)
    turns=defaultdict(list)
    for e in raw:
        if not isinstance(e,dict):
            continue
        sid=str(e.get('speaker_id') or '')
        eid=str(e.get('entity_id') or '')
        vid=entity_to_visual.get(eid)
        # A single legacy ID can designate several DIFFERENT visual tracklets.
        # Resolve the specific face visible at the ASD evidence timestamp.
        if entity_timeline is not None:
            ts=e.get('timestamp')
            try: ts=float(ts)
            except (TypeError,ValueError):
                rejected['missing_asd_timestamp']+=1
                continue
            local=entity_timeline.get(eid) or []
            if not local:
                rejected['entity_not_visible_at_asd_time']+=1
                continue
            neighbors=sorted(((abs(ts-t),v) for t,v in local),key=lambda x:x[0])[:3]
            if not neighbors or neighbors[0][0]>max_time_gap:
                rejected['entity_not_visible_at_asd_time']+=1
                continue
            # Two equally close but incompatible tracklets -> reject.
            if len(neighbors)>1 and neighbors[1][1]!=neighbors[0][1] and abs(neighbors[1][0]-neighbors[0][0])<0.12:
                rejected['ambiguous_temporal_visual_match']+=1
                continue
            vid=neighbors[0][1]
        tid=str(e.get('turn_id') or e.get('segment_id') or '')
        if not vid or not sid or not tid:
            rejected['missing_voice_visual_or_turn'] += 1
            continue
        try:
            score=float(e.get('raw_score',float('-inf')))
            coverage=float(e.get('face_coverage') or 0.)
            face_frames=int(e.get('face_frames') or 0)
            runner=e.get('runner_up_raw_score')
            margin=score-float(runner) if runner is not None else None
        except (ValueError,TypeError):
            rejected['invalid_measurement'] += 1
            continue
        if not (isfinite(score) and isfinite(coverage)):
            rejected['invalid_measurement'] += 1
            continue
        if (score<min_raw_score or coverage<min_face_coverage or
                face_frames<min_face_frames or
                (margin is not None and margin<min_runner_margin)):
            rejected['weak_asd_evidence'] += 1
            continue
        turns[(sid,tid)].append({'speaker_id':sid,'turn_id':tid, 'visual_id':vid,
                                 'source_entity_id':eid,'score':score,
                                 'timestamp':e.get('timestamp'),
                                 'coverage':coverage, 'face_frames':face_frames})
    votes=defaultdict(lambda:defaultdict(list))
    for (sid,tid),items in turns.items():
        best_by_visual={}
        for item in items:
            vid=item['visual_id']
            if vid not in best_by_visual or item['score']>best_by_visual[vid]['score']:
                best_by_visual[vid]=item
        ranked=sorted(best_by_visual.values(),key=lambda x:-x['score'])
        if len(ranked)>1 and ranked[0]['score']-ranked[1]['score']<min_runner_margin:
            rejected['multiple_active_candidates_in_turn']+=1
            continue
        votes[sid][ranked[0]['visual_id']].append(ranked[0])
    links={}
    for sid in sorted(set(voice_nodes)|set(votes)):
        options=votes.get(sid,{})
        ranked=sorted(options.items(), key=lambda kv:(-len(kv[1]),
                -sum(x['score'] for x in kv[1])/len(kv[1]),kv[0]))
        if not ranked:
            links[sid]={'speaker_id':sid,'visual_id':None,'candidate_visual_id':None,
                        'entity_id':None,'candidate_entity_id':None,
                        'status':'no_visual_evidence','verified':False,'evidence_count':0,
                        'support_fraction':0., 'method':'light_asd_temporal'}
            continue
        vid, support=ranked[0]
        total=len({i['turn_id'] for l in options.values() for i in l})
        fraction=len(support)/max(1,total)
        competitor = ranked[1] if len(ranked)>1 else None
        ambiguous=bool(competitor and len(competitor[1])>=len(support)-1 and
                (len(competitor[1])==len(support) or fraction<0.85))
        strong=len(support)>=min_turns and fraction>=min_support_fraction and not ambiguous
        eid=next((alias for alias in visual_nodes[vid]['entity_ids']
                  if entity_to_visual.get(alias)==vid),None)
        links[sid]={
            'speaker_id':sid,
            'visual_id':vid if strong else None,
            'candidate_visual_id':vid,
            'entity_id':eid if strong else None,
            'candidate_entity_id':eid or support[0].get('source_entity_id'),
            'status':'verified_av' if strong else ('conflict' if ambiguous else 'candidate'),
            'verified':strong,
            'evidence_count':len(support),
            'support_fraction':round(fraction,4),
            'turn_ids':sorted(i['turn_id'] for i in support),
            'evidence_timestamps':[x['timestamp'] for x in support],
            'raw_score_mean':round(sum(i['score'] for i in support)/len(support),4),
            'score_type':'uncalibrated_asd_logit',
            'method':'light_asd_temporal',
        }
    # No two different voice IDs are silently assigned the same visual person.
    competing=defaultdict(list)
    for sid,l in links.items():
        if l['verified']:
            competing[l['visual_id']].append(sid)
    for vid,sids in competing.items():
        if len(sids)>1:
            for sid in sids:
                links[sid].update({'visual_id':None,'entity_id':None,
                                   'verified':False,'status':'conflict',
                                   'reason':'multiple_voice_ids_for_one_visual_identity',
                                   'competing_speaker_ids':[s for s in sids if s!=sid]})
    # Explicit, externally reviewed speaker↔entity mapping only.
    for sid,source in (manual_verified_map or {}).items():
        eid=source if isinstance(source,str) else (source.get('visual_id') or source.get('entity_id') if isinstance(source,dict) else None)
        if not eid:
            raise ValueError(f'Mapa revisado sem identidade visual: {sid}')
        eid=str(eid)
        if eid in visual_nodes:
            vid=eid
            eid=next((alias for alias in visual_nodes[vid]['entity_ids']
                      if entity_to_visual.get(alias)==vid),None)
        elif eid in entity_to_visual:
            vid=entity_to_visual[eid]
        else:
            raise ValueError(f'Vinculo revisado desconhecido ou ambiguo: {sid} -> {eid}. Use visual_ID.')
        if isinstance(source,dict) and source.get('verified') is not True:
            raise ValueError('Mapa externo por objeto exige verified=true; nao aceitar candidatos como revisados.')
        # A human must explicitly attest any deliberate override.
        links[str(sid)]={'speaker_id':str(sid),'visual_id':vid,'candidate_visual_id':vid,
                         'entity_id':str(eid) if eid else None,'candidate_entity_id':str(eid) if eid else None,
                         'status':'human_verified','verified':True,
                         'evidence_count':0,'support_fraction':None,
                         'method':'human_review','score_type':None}
    # Apply global uniqueness after manual overrides, too. Do not override one
    # validated human decision with an unreviewed automatic assignment.
    competing=defaultdict(list)
    for sid,l in links.items():
        if l['verified']:
            competing[l['visual_id']].append(sid)
    for vid,sids in competing.items():
        if len(sids)<2:
            continue
        human=[s for s in sids if links[s]['status']=='human_verified']
        for sid in sids:
            if len(human)==1 and sid==human[0]:
                continue
            links[sid]['status']='conflict'
            links[sid]['verified']=False
            links[sid]['entity_id']=None
            links[sid]['visual_id']=None
            links[sid]['reason']='ambiguous_voice_to_visual_ownership'
    legacy_map={sid:{'entity_id':l['entity_id'],'verified':True,'method':l['method']}
                for sid,l in links.items() if l['verified'] and l['entity_id']}
    return {'links':links, 'speaker_entity_map':legacy_map,
            'status_counts':{k:sum(x['status']==k for x in links.values())
                             for k in sorted({x['status'] for x in links.values()})},
            'rejected':dict(rejected),
            'raw_observation_count':len(raw)}
