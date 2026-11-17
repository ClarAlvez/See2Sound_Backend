"""Resolve nomes mencionados no audio sem inventar identidade de falante.

O WhisperPauseDetector fornece texto+intervalos, NAO diarizacao ou lip sync.
Portanto nomes sem vinculo verificavel ficam como hipotese nao-confirmada.
"""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Any

from ai.spectra.Correlation.requesters.transcription import TranscriptionRequester

# v0.8: apenas padroes ancorados. Uma palavra antes de virgula NAO e nome.
WORD = r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]{1,23}"
NAME = rf"(?P<name>{WORD}(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'’-]{{1,23}})?)"
PATTERNS = (
    ('self_introduction', re.compile(
        rf"(?i:\b(?:meu nome é|meu nome e|me chamo|eu me chamo|eu sou|sou|my name is|i am|i'm|me llamo)\s+){NAME}")),
    ('introduction', re.compile(
        rf"(?i:\b(?:esta é a|essa é a|este é o|esse é o|este é|esta é|o nome dela é|o nome dele é|ela se chama|ele se chama|this is|te apresento|presento a|presento al)\s+){NAME}")),
    ('address', re.compile(
        rf"(?i:\b(?:oi|olá|ei|hey|hello|hi)\s*,?\s+){NAME}(?=\s*[,!?.]|$)")),
)
# Evita falsos positivos frequentes de ASR em frases comuns.
REJECT = set("""
    aqui ali hoje agora oi olá ei amigo amiga pessoal senhor senhora senhores
    obrigado obrigada tudo bem voce você vocês ele ela eles elas nos nós
    feliz forte triste pronto pronta cansado cansada ninguém todos todas
    não sim isso isto aquela aquele sou eu um uma uns umas o a os as the
    i you here there fine good sorry hello hey hi friend sir miss this that
    happy me my name your old ready thanks please vamos venha então entao
    claramente resumindo coisa coisas aprender cérebro cerebro mão mao verdade
    bom boa ok certo certa estranho estranha agora depois antes quando
    porque pois contudo entretanto até ate logo nada nunca sempre mesmo
    pode entrar acho olha vejam sabe sabem são sao fui foi será sera era
    tenho tem temos está esta estão estao estamos fazer falando dizem disse
    meu minha meus suas seu seus nossa nosso professor doutor nome cara
    senhorita doutora gente ainda muito pouco outro outra um dois três tres
    mas porém porem só so realmente certamente obviamente talvez
    pelo pela para por com sem dentro fora amanha amanhã também tambem
""".split())


def normalized_name(name: str) -> str | None:
    words = re.sub(r'\s+', ' ', str(name).strip(" .,;:!?\"'‘’")).split()
    if not words or len(words) > 2 or any(len(w) < 2 or len(w) > 24 for w in words):
        return None
    if any(w.casefold() in REJECT for w in words):
        return None
    if not all(re.fullmatch(WORD, w) for w in words):
        return None
    return ' '.join(w[:1].upper() + w[1:] for w in words)


def detect_mentions(text: str, *, known_names=()):
    """Reconhece apresentacoes e vocativos ancorados, nunca qualquer 'X,'."""
    matches = []
    seen_spans = set()
    raw = text or ''

    def add(kind, result, name):
        span = [result.start('name'), result.end('name')]
        signature = (span[0], span[1], kind)
        if signature not in seen_spans:
            matches.append({'name': name, 'kind': kind, 'span': span})
            seen_spans.add(signature)

    for kind, pattern in PATTERNS:
        for result in pattern.finditer(raw):
            name = normalized_name(result.group('name'))
            if name:
                add(kind, result, name)

    # Palavra isolada + exclamacao pode ser vocativo, mas sem alvo visual certo.
    bare = re.fullmatch(rf'\s*{NAME}[!?]+\s*', raw)
    if bare:
        name = normalized_name(bare.group('name'))
        if name and bare.group('name')[0].isupper():
            add('address', bare, name)

    # Repeticoes de nomes ja introduzidos podem ser registradas como vocativos.
    # Isso NAO confirma identidade de quem esta falando.
    for anchored in sorted(set(known_names), key=len, reverse=True):
        name = normalized_name(anchored)
        if not name:
            continue
        pattern = re.compile(rf'(?<!\w)(?P<name>{re.escape(name)})\s*[,!](?=\s|$)', re.I)
        for match in pattern.finditer(raw):
            if any(x['span'] == [match.start('name'), match.end('name')] for x in matches):
                continue
            add('address', match, name)
        # Referencias narrativas: "Eleanor decidiu..." nao sao vocativos.
        # Apenas registrar a mencao; a pessoa nomeada pode nem estar na tela.
        any_occurrence = re.compile(rf'(?<!\w)(?P<name>{re.escape(name)})(?!\w)', re.I)
        for match in any_occurrence.finditer(raw):
            if any(x['span'] == [match.start('name'), match.end('name')] for x in matches):
                continue
            add('reference', match, name)
    return matches


def anchored_names(segments):
    """Somente nomes ancorados por autoapresentacao ou apresentacao explicita."""
    return {mention['name'] for segment in segments
            for mention in detect_mentions(segment.get('text', ''))
            if mention['kind'] in {'self_introduction', 'introduction'}}


def canonical(entity_id, remap):
    seen = set()
    while entity_id in remap and entity_id not in seen:
        seen.add(entity_id)
        entity_id = remap[entity_id]
    return entity_id


def span_overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


class SpeechIdentityResolver:
    """Cruza transcricao, frames rastreados e intervalos de cena.

    Confirmacao automatica so com grounding VERIFICADO: mapeamento manual
    speaker_id->entity_id ou hint externo explicitamente validado. Sem isso,
    anota candidatos mas nunca transforma uma coincidencia em nome certo.
    """
    def __init__(self, config=None):
        self.config = config
        self.requester = TranscriptionRequester()

    def _window(self):
        return float(getattr(self.config, 'speech_frame_tolerance_seconds', 1.0))

    def _scene_intervals(self, payload):
        if not isinstance(payload, dict):
            return []
        spectra = payload.get('spectra') or {}
        outputs = spectra.get('outputs', []) if isinstance(spectra, dict) else []
        if not outputs and isinstance(payload.get('raw_frame_results'), list):
            outputs = [payload]
        intervals = []
        for i, item in enumerate(outputs):
            if not isinstance(item, dict):
                continue
            context = item.get('context') or {}
            ordinal = context.get('scene_index', item.get('scene_index', i))
            scene_id = f'scene_{int(ordinal):04d}' if str(ordinal).isdigit() else str(ordinal)
            try:
                start, end = float(item.get('start_time')), float(item.get('end_time'))
            except (TypeError, ValueError):
                continue
            if end >= start:
                intervals.append((start, end, scene_id))
        return intervals

    @staticmethod
    def _speaker_link(speaker_id, verified_map, remap):
        if not speaker_id or not isinstance(verified_map, dict):
            return None
        link = verified_map.get(str(speaker_id))
        if isinstance(link, str):
            # Mapa fornecido pelo usuario como referencia verificada.
            return {'entity_id': canonical(link, remap), 'verified': True, 'source': 'verified_speaker_map'}
        if isinstance(link, dict) and link.get('entity_id'):
            return {'entity_id': canonical(str(link['entity_id']), remap),
                    'verified': link.get('verified') is True,
                    'source': str(link.get('method') or 'external_speaker_map')}
        return None

    def _frame_context(self, seg, frames, intervals):
        start, end = seg['start'], seg['end']
        # A consulta visual e aproximada: Whisper nao fornece word timestamps
        # nesse pipeline e os frames nao sao necessariamente frequentes.
        tolerance = self._window()
        nearby = [f for f in frames if start - tolerance <= float(f['timestamp']) <= end + tolerance]
        selected = sorted(nearby, key=lambda f: min(abs(float(f['timestamp']) - start),
                                                         abs(float(f['timestamp']) - end)))
        # Para falas curtas, nao usar pessoas que surgiram muito depois.
        selected = selected[:min(20, len(selected))]
        ids = set()
        talking = set()
        scenes = set()
        frame_ids = set()
        for frame in selected:
            frame_ids.add(frame.get('frame_id'))
            scenes.add(str(frame.get('scene_id', 'scene_0001')))
            ids.update(p['entity_id'] for p in frame.get('people', []) if p.get('entity_id'))
            for action in frame.get('actions', []) or []:
                if not isinstance(action, dict) or not action.get('entity_id'):
                    continue
                if any(p.get('label') == 'talking' and float(p.get('score', 0)) >= 0.55
                       for p in action.get('predictions', [])):
                    talking.add(action['entity_id'])
        overlapping = [s for t0, t1, s in intervals if span_overlap(start, end, t0, t1) > 0
                       or (start == end and t0 <= start <= t1)]
        if overlapping:
            scenes = set(overlapping)
            # Nao puxar o personagem de outra cena so porque o frame caiu perto do corte.
            scene_frames = [f for f in selected if str(f.get('scene_id')) in scenes]
            if scene_frames:
                ids = {p['entity_id'] for f in scene_frames for p in f.get('people', [])
                       if p.get('entity_id')}
                frame_ids = {f.get('frame_id') for f in scene_frames}
                talking = {a['entity_id'] for f in scene_frames for a in f.get('actions', []) or []
                           if isinstance(a, dict) and a.get('entity_id') and any(
                               pred.get('label') == 'talking' and float(pred.get('score', 0)) >= 0.55
                               for pred in a.get('predictions', []))}
        return {
            'scene_ids': sorted(scenes),
            'frame_ids': sorted(x for x in frame_ids if x is not None),
            'visible_entity_ids': sorted(ids),
            'talking_entity_ids': sorted(talking & ids),
            'timing_precision': 'segment',
        }

    def _choose(self, mention, segment, context, speaker_link, hints, entities):
        kind, name = mention['kind'], mention['name']
        speaker_entity = speaker_link['entity_id'] if speaker_link else None
        target = None
        score = 0.0
        verified = False
        evidence = ''
        # Hints externos sao associados ao segmento e nome exatos, nunca ao
        # video inteiro. Aceita revisao humana ou lip-sync ja validado.
        for hint in hints:
            if (not isinstance(hint, dict) or str(hint.get('segment_id')) != segment['segment_id']
                    or str(hint.get('name', '')).casefold() != name.casefold()):
                continue
            eid = hint.get('entity_id')
            if not eid or eid not in entities:
                continue
            target = eid
            verified = hint.get('verified') is True
            evidence = str(hint.get('method') or 'external_hint')
            score = 0.99 if verified else 0.65
            break
        if target is None and kind == 'self_introduction':
            if speaker_entity in entities and speaker_link.get('verified'):
                target, verified, score, evidence = speaker_entity, True, 0.98, 'verified_speaker_self_introduction'
            elif speaker_entity in entities:
                target, score, evidence = speaker_entity, 0.72, 'unverified_speaker_mapping'
            elif len(context['visible_entity_ids']) == 1:
                target = context['visible_entity_ids'][0]
                score = 0.55 if target in context['talking_entity_ids'] else 0.38
                evidence = 'single_visible_person_unverified_voice'
        elif target is None and kind in ('introduction', 'address'):
            # Nome pode apontar a quem e apresentado/chamado, NUNCA ao
            # falante por padrao. Quando existem varios, nao escolher.
            candidates = [x for x in context['visible_entity_ids'] if x != speaker_entity]
            if len(candidates) == 1:
                target = candidates[0]
                score = 0.66 if kind == 'introduction' else 0.52
                evidence = 'single_possible_addressee_unverified'
        if target not in entities:
            target, score, verified, evidence = None, 0.0, False, 'insufficient_grounding'
        return {'entity_id': target, 'heuristic_score': round(score, 4),
                'verified': verified, 'evidence_type': evidence}

    def resolve(self, transcription, frames, entities, *, pipeline_payload=None,
                speaker_entity_map=None, entity_hints=None, entity_remap=None):
        transcript = self.requester.request(transcription)
        remap = entity_remap or {}
        verified_map = speaker_entity_map or {}
        hints = []
        for hint in entity_hints or []:
            if isinstance(hint, dict):
                hint = dict(hint)
                if hint.get('entity_id'):
                    hint['entity_id'] = canonical(hint['entity_id'], remap)
                hints.append(hint)
        intervals = self._scene_intervals(pipeline_payload)
        timelines = []
        evidence_per_entity = defaultdict(lambda: defaultdict(list))
        unassigned = []
        known_names = anchored_names(transcript['segments'])
        for segment in transcript['segments']:
            context = self._frame_context(segment, frames, intervals)
            speaker_link = self._speaker_link(segment['speaker_id'], verified_map, remap)
            mentions = []
            for mention in detect_mentions(segment['text'], known_names=known_names):
                assignment = self._choose(mention, segment, context, speaker_link, hints, entities)
                evidence = {**mention, **assignment, 'segment_id': segment['segment_id'],
                            'timestamp': round((segment['start'] + segment['end']) / 2, 3),
                            'scene_ids': context['scene_ids']}
                mentions.append(evidence)
                if assignment['entity_id'] is not None:
                    evidence_per_entity[assignment['entity_id']][mention['name']].append(evidence)
                else:
                    unassigned.append(evidence)
            timelines.append({**segment, **context,
                              'speaker_entity_id': speaker_link['entity_id'] if speaker_link else None,
                              'speaker_grounding_verified': bool(speaker_link and speaker_link['verified']),
                              'name_mentions': mentions})
        candidates = {}
        confirmed_names = {}
        for entity_id, by_name in evidence_per_entity.items():
            candidates[entity_id] = []
            for name, evidences in by_name.items():
                verified_count = sum(1 for e in evidences if e['verified'])
                score = max(e['heuristic_score'] for e in evidences)
                candidates[entity_id].append({
                    'name': name, 'status': 'confirmed' if verified_count else 'candidate',
                    'heuristic_score': round(score, 4),
                    'mention_count': len(evidences), 'verified_evidence_count': verified_count,
                    'segment_ids': sorted(set(e['segment_id'] for e in evidences)),
                    'evidence_types': sorted(set(e['evidence_type'] for e in evidences)),
                })
            candidates[entity_id].sort(key=lambda x: (-x['verified_evidence_count'], -x['heuristic_score'], x['name']))
            # Dois nomes diferentes com evidencia verificada: conflito,
            # nao supor qual e o nome principal nem descartar apelidos.
            confirmed = [x for x in candidates[entity_id] if x['verified_evidence_count'] > 0]
            if len(confirmed) == 1:
                confirmed_names[entity_id] = confirmed[0]['name']
        # Dois personagens podem ter mesmo nome. Evitar promocao nao-verificada.
        inverse = defaultdict(list)
        for eid, name in confirmed_names.items():
            inverse[name.casefold()].append(eid)
        ambiguous = set(eid for ids in inverse.values() if len(ids) > 1 for eid in ids)
        for eid in ambiguous:
            confirmed_names.pop(eid, None)
        dialogue_by_scene = defaultdict(list)
        for segment in timelines:
            entry = {'segment_id': segment['segment_id'], 'start': segment['start'],
                     'end': segment['end'], 'text': segment['text'],
                     'speaker_entity_id': segment['speaker_entity_id'],
                     'name_mentions': segment['name_mentions']}
            for scene_id in segment['scene_ids']:
                dialogue_by_scene[scene_id].append(entry)
        return {
            'source': 'audio.speech_analysis.speech_segments',
            'available': transcript['available'],
            'language': transcript['language'],
            'diarization_available': transcript['diarization_available'],
            'speaker_mapping_provided': bool(verified_map),
            'timing_precision': 'segment',
            'segments': timelines,
            'dialogue_by_scene': dict(dialogue_by_scene),
            'name_candidates': candidates,
            'unassigned_name_mentions': unassigned,
            'confirmed_names': confirmed_names,
            'warnings': (["Whisper atual nao fornece diarizacao; nomes sem vinculo de voz verificado permanecem candidatos."]
                         if transcript['available'] and not transcript['diarization_available'] else []),
        }
