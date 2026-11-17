"""Evidence-accumulating speaker -> person links.

Scores returned by Light-ASD are logits, NOT calibrated probabilities. Links
require independent speaking turns, visual coverage and competition checks.
"""
from __future__ import annotations
from collections import defaultdict


class SpeakerLinkMemory:
    def __init__(self, *, min_turns=2, min_raw_score=0.0, min_turn_margin=0.3,
                 min_support_fraction=0.7, min_face_coverage=0.5, min_face_frames=12,
                 min_entity_margin=0.5, allow_automatic_name_confirmation=False):
        self.min_turns = min_turns
        self.min_raw_score = min_raw_score
        self.min_turn_margin = min_turn_margin
        self.min_support_fraction = min_support_fraction
        self.min_face_coverage = min_face_coverage
        self.min_face_frames = min_face_frames
        self.min_entity_margin = min_entity_margin
        self.allow_automatic_name_confirmation = allow_automatic_name_confirmation

    def resolve(self, evidence, valid_entity_ids=None, remap=None):
        remap = remap or {}
        allowed = set(valid_entity_ids or []) if valid_entity_ids is not None else None
        def canonical(entity):
            visited = set()
            while entity in remap and entity not in visited:
                visited.add(entity)
                entity = remap[entity]
            return entity
        # One independent vote per turn / entity. Discard weak raw logits.
        by_speaker = defaultdict(lambda: defaultdict(dict))
        diagnostics = []
        for item in evidence or []:
            if not isinstance(item, dict):
                continue
            sid = item.get('speaker_id')
            eid = item.get('entity_id')
            turn = item.get('turn_id') or item.get('segment_id')
            if not sid or not eid or turn is None:
                continue
            eid = canonical(str(eid))
            if allowed is not None and eid not in allowed:
                continue
            try:
                score = float(item.get('raw_score', -999.0))
                runner_value = item.get('runner_up_raw_score')
                runner = float(runner_value) if runner_value is not None else -999.0
                coverage = float(item.get('face_coverage', 0.0))
                frames = int(item.get('face_frames', 0))
            except (TypeError, ValueError):
                continue
            if score < self.min_raw_score or coverage < self.min_face_coverage or frames < self.min_face_frames:
                diagnostics.append({'turn_id': str(turn), 'speaker_id': sid, 'entity_id': eid,
                                    'decision': 'insufficient_active_speaker_evidence'})
                continue
            # Runner-up can be genuinely absent (only one visible face).
            margin = score - runner if runner != -999.0 else None
            if margin is not None and margin < self.min_turn_margin:
                diagnostics.append({'turn_id': str(turn), 'speaker_id': sid, 'entity_id': eid,
                                    'decision': 'ambiguous_active_faces', 'raw_margin': round(margin,4)})
                continue
            previous = by_speaker[str(sid)][eid].get(str(turn))
            if previous is None or previous['raw_score'] < score:
                by_speaker[str(sid)][eid][str(turn)] = {
                    'raw_score': score, 'face_coverage': coverage,
                    'face_frames': frames, 'turn_id': str(turn),
                    'timestamp': item.get('timestamp'),
                }
        links = {}
        for sid, by_entity in sorted(by_speaker.items()):
            total_unique_turns = len(set(tid for turns in by_entity.values() for tid in turns))
            ranked = sorted(by_entity.items(), key=lambda kv:(-len(kv[1]),
                          -sum(x['raw_score'] for x in kv[1].values())/len(kv[1]), kv[0]))
            if not ranked:
                continue
            eid, supporting = ranked[0]
            count = len(supporting)
            fraction = count/max(total_unique_turns,1)
            rival = ranked[1] if len(ranked)>1 else None
            # Do not accept two identities with near-equal support; require margin
            # in raw mean-logit when counts tie.
            rival_ambiguous = (rival is not None and len(rival[1]) == count and
                               (sum(x['raw_score'] for x in supporting.values())/count -
                                sum(x['raw_score'] for x in rival[1].values())/len(rival[1])) < self.min_entity_margin)
            strong = (count >= self.min_turns and fraction >= self.min_support_fraction
                      and not rival_ambiguous)
            links[sid] = {
                'speaker_id': sid,
                'entity_id': eid if strong else None,
                'candidate_entity_id': eid,
                'status': 'high_confidence' if strong else 'candidate',
                'evidence_count': count,
                'support_fraction': round(fraction,4),
                'raw_score_mean': round(sum(x['raw_score'] for x in supporting.values())/count,4),
                'turn_ids': sorted(supporting),
                'verified': bool(strong and self.allow_automatic_name_confirmation),
                'method': 'light_asd_multi_turn',
                'score_type': 'uncalibrated_logit',
            }
        # Global uniqueness: two speakers must not be assigned to one visible
        # person in the same video without further voice-identity evidence.
        reverse = defaultdict(list)
        for sid, link in links.items():
            if link['entity_id'] is not None:
                reverse[link['entity_id']].append(sid)
        for eid, speakers in reverse.items():
            if len(speakers) <= 1:
                continue
            # Sem verificacao de identidade vocal, a mesma pessoa nao deve
            # receber duas vozes. Dominancia MUITO forte pode manter um
            # falante e rejeitar os demais; empate continua sem nome.
            ranked_speakers = sorted(speakers, key=lambda sid:(
                -links[sid]['evidence_count'], -links[sid]['raw_score_mean'], sid))
            winner, runner_up = ranked_speakers[:2]
            win, rival = links[winner], links[runner_up]
            dominant = (
                win['evidence_count'] >= max(6, 3 * rival['evidence_count'])
                and win['support_fraction'] >= 0.85
                and win['raw_score_mean'] >= rival['raw_score_mean'] + 0.25
            )
            for sid in speakers:
                links[sid]['competing_speaker_ids'] = sorted(s for s in speakers if s != sid)
                if dominant and sid == winner:
                    links[sid]['competition_resolved'] = 'dominant_independent_evidence'
                    continue
                links[sid]['entity_id'] = None
                links[sid]['verified'] = False
                links[sid]['status'] = 'conflict'
                links[sid]['reason'] = ('dominated_by_stronger_speaker'
                                        if dominant else 'multiple_speakers_for_same_entity')
        mapping = {sid: {'entity_id': link['entity_id'], 'verified': link['verified'],
                         'method': link['method']} for sid,link in links.items() if link['entity_id']}
        return {'links': links, 'speaker_entity_map': mapping,
                'status_counts': {status: sum(v['status']==status for v in links.values())
                                  for status in sorted({v['status'] for v in links.values()})},
                'raw_observation_count': len(evidence or []), 'rejected': diagnostics}
