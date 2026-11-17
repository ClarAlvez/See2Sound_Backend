"""Voice-first identities: names are speech claims, NOT visual person IDs.

Only explicit self introductions claim the speaker's own name; addressing or
mentioning a person never names the speaker.  Whisper may be mistaken, so every
claim remains auditable and is not silently promoted to a visual identity.
"""
from __future__ import annotations
from collections import defaultdict
import unicodedata

# Common occupations/roles are not character names, even when speech
# transcription erroneously assigns them to a self-introduction pattern.
GENERIC_ROLE_WORDS = {
    'ator', 'atriz', 'pessoa', 'homem', 'mulher', 'professor', 'professora',
    'estudante', 'aluno', 'aluna', 'amigo', 'amiga', 'diretor', 'diretora',
    'narrador', 'narradora', 'senhor', 'senhora', 'cara', 'moco', 'moca',
    'actor', 'actress', 'person', 'man', 'woman', 'teacher', 'student',
}


def is_person_name_candidate(name):
    name=str(name or '').strip()
    if not name or len(name)<2 or len(name)>64 or any(char.isdigit() for char in name):
        return False
    normalized=''.join(c for c in unicodedata.normalize('NFKD',name.casefold())
                       if not unicodedata.combining(c))
    return normalized not in GENERIC_ROLE_WORDS



def build_voice_identities(segments, diarization=None):
    groups = defaultdict(lambda: {'segments': [], 'claims': defaultdict(list), 'mentions': [], 'rejected_claims': []})
    for item in segments or []:
        if not isinstance(item, dict):
            continue
        sid = item.get('speaker_id')
        if sid is None or str(sid).lower() in ('unknown', 'none', ''):
            continue
        sid = str(sid)
        group = groups[sid]
        group['segments'].append(str(item.get('segment_id') or ''))
        for mention in item.get('name_mentions') or []:
            if not isinstance(mention, dict) or not mention.get('name'):
                continue
            name = str(mention['name']).strip()
            entry = {'name': name, 'kind': str(mention.get('kind') or ''),
                     'segment_id': str(item.get('segment_id') or ''),
                     'timestamp': mention.get('timestamp', item.get('start')),
                     'text': str(item.get('text') or '')[:240]}
            group['mentions'].append(entry)
            if mention.get('kind') == 'self_introduction':
                if is_person_name_candidate(name):
                    group['claims'][name.casefold()].append(entry)
                else:
                    group['rejected_claims'].append(entry)
    for sid in (diarization or {}).get('speakers') or []:
        groups[str(sid)]  # preserve diarized speakers who have no transcript
    voices = {}
    for sid, group in sorted(groups.items()):
        claim_groups = group['claims']
        distinct_claims = sorted(claim_groups)
        names = [claim_groups[key][0]['name'] for key in distinct_claims]
        if len(names) == 1:
            label, status = names[0], 'self_declared_unverified'
        elif len(names) > 1:
            label, status = None, 'conflicting_self_introductions'
        else:
            label, status = None, 'unnamed'
        voices[sid] = {
            'voice_id': sid,
            'name_claim': label,
            'name_status': status,
            'name_claims': [{'name': claim_groups[key][0]['name'],
                             'segment_ids': sorted(set(e['segment_id'] for e in claim_groups[key])),
                             'evidence': claim_groups[key]}
                            for key in distinct_claims],
            'segment_count': len(group['segments']),
            'name_mentions': group['mentions'],
            'rejected_self_claims': group['rejected_claims'],
            'visual_id': None,
        }
    return voices
