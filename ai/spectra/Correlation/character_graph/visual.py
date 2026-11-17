"""Conservative cross-scene face clustering for CharacterGraph v1.0.1.

The old scene tracklets remain separate unless face evidence joins them.
Unlike v1.0, nearly identical *multiple* tracklets are not all rejected
because the two best matches have similar scores. Explicit co-occurrence
contradictions and complete-link face consistency are still enforced.
"""
from __future__ import annotations
from collections import defaultdict
from math import sqrt
from pathlib import Path


def cosine(a, b):
    return sum(x*y for x,y in zip(a,b)) / max(1e-10, sqrt(sum(x*x for x in a))*sqrt(sum(y*y for y in b)))


def tracklet_key(frame, person):
    scene=str(frame.get('scene_id') or 'scene_unassigned')
    local=person.get('track_id')
    if local is None:
        local='legacy_'+str(person.get('entity_id'))
    return scene+'|'+str(local)


def resolve_crop(path, root=None):
    if not path:
        return None
    raw=Path(path)
    if raw.is_file():
        return str(raw)
    if root:
        for candidate in (Path(root)/raw, Path(root)/raw.name,
                          Path(root)/'person_crops'/raw.name):
            if candidate.is_file():
                return str(candidate)
    return None


def _spread_crops(records, n, root):
    # Spread samples across a tracklet, rather than choosing only its first
    # frames (which often show an occluded or turned-away face).
    if not records or n <= 0:
        return []
    candidates=[]
    count=len(records)
    probes=min(count, max(n, n*2))
    for i in range(probes):
        pos=min(count-1,int((i+0.5)*count/probes))
        path=resolve_crop(records[pos].get('crop_path'),root)
        if path and path not in candidates:
            candidates.append(path)
    return candidates


def build_visual_identities(frames, entities, *, encoder=None, max_entities=260,
                            max_crops_per_entity=3, min_similarity=0.55,
                            min_margin=0.08, strong_similarity=0.65,
                            min_face_samples=2, crop_root=None, attention_times=()):
    if not (0 <= min_similarity <= strong_similarity <= 1):
        raise ValueError('Esperado 0 <= min_similarity <= strong_similarity <= 1.')
    max_entities=max(0,int(max_entities))
    max_crops_per_entity=max(1,int(max_crops_per_entity))
    min_face_samples=max(1,int(min_face_samples))
    tracklets=defaultdict(list)
    forbidden=defaultdict(set)
    alias_ids=defaultdict(set)
    for frame in frames or []:
        ts=float(frame.get('timestamp',0.0))
        keys=set()
        for p in frame.get('people') or []:
            eid=p.get('entity_id')
            if not eid:
                continue
            eid=str(eid)
            key=tracklet_key(frame,p)
            keys.add(key)
            alias_ids[key].add(eid)
            tracklets[key].append({'timestamp':ts,'scene_id':frame.get('scene_id'),
                                   'crop_path':p.get('crop_path'),'entity_id':eid})
        for key in keys:
            forbidden[key].update(keys-{key})
    ids=sorted(tracklets)
    attention=[float(t) for t in (attention_times or []) if t is not None]
    alias_scenes=defaultdict(set)
    for key, rows in tracklets.items():
        for eid in alias_ids[key]:
            alias_scenes[eid].add(str(rows[0]['scene_id']))

    def priority(key):
        times=[r['timestamp'] for r in tracklets[key]]
        anchored=any(abs(t-v)<=8.0 for t in times for v in attention)
        # Frequent IDs are merely a sampling hint; they are NEVER merged on
        # legacy ID similarity. Some legacy IDs represent different people.
        recurrence=max((len(alias_scenes[eid]) for eid in alias_ids[key]),default=0)
        return (-int(anchored),-min(100,recurrence),-len(tracklets[key]),key)

    sampled=sorted(ids,key=priority)[:max_entities]
    features={}
    stats={'face_encoder_enabled':encoder is not None,'sampled_crops':0,
           'valid_face_embeddings':0,'face_clusters_merged':0,
           'tracklets':len(ids),'sampled_tracklets':len(sampled) if encoder is not None else 0,
           'skipped_tracklets_due_to_budget':max(0,len(ids)-len(sampled)) if encoder is not None else 0,
           'eligible_tracklets':0,'candidate_pairs_above_threshold':0,
           'strong_candidate_pairs':0,'weak_mutual_candidate_pairs':0,
           'rejected_cooccurrence_pairs':0,'rejected_complete_link_pairs':0,
           'rejected_weak_ambiguous_pairs':0,
           'face_matching_policy':'strong_or_mutual_with_complete_link_v101'}
    if encoder is not None:
        for key in sampled:
            paths=_spread_crops(tracklets[key],max_crops_per_entity,crop_root)
            vectors=[]
            for path in paths:
                stats['sampled_crops']+=1
                vector=encoder.encode(path)
                if vector is None:
                    continue
                if not vectors or len(vector)==len(vectors[0]):
                    vectors.append(vector)
                    stats['valid_face_embeddings']+=1
                if len(vectors)>=max_crops_per_entity:
                    break
            if vectors:
                summed=[sum(v[i] for v in vectors) for i in range(len(vectors[0]))]
                norm=sqrt(sum(x*x for x in summed))
                if norm>1e-10:
                    features[key]={'embedding':[x/norm for x in summed],
                                   'samples':len(vectors)}
    eligible=sorted(k for k in features if features[k]['samples']>=min_face_samples)
    stats['eligible_tracklets']=len(eligible)
    parent={k:k for k in ids}
    members={k:{k} for k in ids}

    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]]
            k=parent[k]
        return k

    pair_scores={}
    ranked=defaultdict(list)
    for i,a in enumerate(eligible):
        for b in eligible[i+1:]:
            if b in forbidden[a]:
                stats['rejected_cooccurrence_pairs']+=1
                continue
            score=cosine(features[a]['embedding'],features[b]['embedding'])
            if score<min_similarity:
                continue
            pair_scores[(a,b)]=score
            ranked[a].append((score,b))
            ranked[b].append((score,a))
    stats['candidate_pairs_above_threshold']=len(pair_scores)
    for key in ranked:
        ranked[key].sort(key=lambda row:(-row[0],row[1]))

    def reciprocal_unique(a,b,score):
        # Weak matches must be each other's best choices *with* a margin.
        # Strong matches do not need global uniqueness: several tracklets of
        # the same real face would otherwise block one another forever.
        for current, other in ((a,b),(b,a)):
            options=ranked[current]
            if not options or options[0][1]!=other:
                return False
            second=options[1][0] if len(options)>1 else -1.0
            if score-second<min_margin:
                return False
        return True

    edges=[]
    for (a,b),score in pair_scores.items():
        if score>=strong_similarity:
            stats['strong_candidate_pairs']+=1
            edges.append((score,a,b))
        elif reciprocal_unique(a,b,score):
            stats['weak_mutual_candidate_pairs']+=1
            edges.append((score,a,b))
        else:
            stats['rejected_weak_ambiguous_pairs']+=1
    edges.sort(key=lambda row:(-row[0],row[1],row[2]))

    def compatible(left,right):
        for a in members[left]:
            for b in members[right]:
                if b in forbidden[a]:
                    return False,'cooccurrence'
                # Complete-link: every matched representative within both
                # clusters must be compatible, preventing weak transitive
                # chains (A~B, B~C but A!=C).
                if a in features and b in features:
                    if cosine(features[a]['embedding'],features[b]['embedding'])<min_similarity:
                        return False,'complete_link'
        return True,None

    for score,a,b in edges:
        ra,rb=find(a),find(b)
        if ra==rb:
            continue
        ok,reason=compatible(ra,rb)
        if not ok:
            stat='rejected_cooccurrence_pairs' if reason=='cooccurrence' else 'rejected_complete_link_pairs'
            stats[stat]+=1
            continue
        if len(members[ra])<len(members[rb]):
            ra,rb=rb,ra
        parent[rb]=ra
        members[ra].update(members.pop(rb))
        stats['face_clusters_merged']+=1

    clusters=defaultdict(list)
    for key in ids:
        clusters[find(key)].append(key)
    visual={}
    tracklet_to_visual={}
    for index,members_list in enumerate(sorted(clusters.values(),key=lambda x:min(x)),1):
        vid=f'visual_{index:04d}'
        rows=[r for key in members_list for r in tracklets[key]]
        aliases=sorted({eid for key in members_list for eid in alias_ids[key]})
        samples=sum(features.get(key,{}).get('samples',0) for key in members_list)
        visual[vid]={
            'visual_id':vid,'tracklet_ids':sorted(members_list),'entity_ids':aliases,
            'first_seen':round(min(r['timestamp'] for r in rows),3),
            'last_seen':round(max(r['timestamp'] for r in rows),3),
            'observations':len(rows),
            'scene_ids':sorted({str(r['scene_id']) for r in rows if r['scene_id'] is not None}),
            'face_samples':samples,'resolved_name':None,
            'identity_method':('sface_complete_link' if len(members_list)>1 else
                               'face_sampled_unmerged' if samples else 'local_tracklet_unverified')}
        for key in members_list:
            tracklet_to_visual[key]=vid
    alias_to_vids=defaultdict(set)
    timeline=defaultdict(list)
    for key,rows in tracklets.items():
        vid=tracklet_to_visual[key]
        for row in rows:
            eid=row['entity_id']
            alias_to_vids[eid].add(vid)
            timeline[eid].append((row['timestamp'],vid))
    unique_alias={eid:next(iter(vids)) for eid,vids in alias_to_vids.items() if len(vids)==1}
    for eid,entries in timeline.items():
        entries.sort(key=lambda row:row[0])
    stats['legacy_ids_split_across_visuals']=sum(len(v)>1 for v in alias_to_vids.values())
    return visual,unique_alias,stats,tracklet_to_visual,dict(timeline)
