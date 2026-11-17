"""See2Sound Narrative 0.7: verbalização ancorada na Correlation 0.6.

Não faz inferência nem usa transcrição como descrição visual. Preserva o
LLMNarrativeGenerator: delta, filtros, reescrita Qwen e contratos do Voice.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any, Dict, List

from ai.narrative.narrative_generator import LLMNarrativeGenerator


SUBJECTS = {'woman': 'mulher', 'man': 'homem', 'person': 'pessoa'}
PERSON_ATTRS = {
    'long_hair': 'de cabelos longos',
    'short_hair': 'de cabelos curtos',
    'black_clothes': 'com roupas pretas',
    'white_clothes': 'com roupas brancas',
    'gray_clothes': 'com roupas cinzas',
    'blue_clothes': 'com roupas azuis',
    'red_clothes': 'com roupas vermelhas',
    'dress': 'de vestido',
    'glasses': 'de óculos',
}
ACTIONS = {
    'walking': 'caminha', 'running': 'corre', 'cycling': 'anda de bicicleta',
    'sitting': 'está sentada', 'standing': 'está em pé',
    'dancing': 'dança', 'reading': 'lê', 'writing': 'escreve',
    'drinking': 'bebe', 'eating': 'come', 'cooking': 'cozinha',
    'phone_use': 'usa um telefone', 'computer_use': 'usa um computador',
    'talking': 'fala', 'swimming': 'nada', 'driving': 'dirige',
    'cleaning': 'limpa', 'playing': 'brinca',
}
OBJECTS = {
    'backpack': ('uma mochila', 'mochila'), 'bag': ('uma bolsa', 'bolsa'),
    'handbag': ('uma bolsa', 'bolsa'), 'umbrella': ('um guarda-chuva', 'guarda-chuva'),
    'phone': ('um telefone', 'telefone'), 'book': ('um livro', 'livro'),
    'cup': ('um copo', 'copo'), 'bottle': ('uma garrafa', 'garrafa'),
    'glasses': ('óculos', 'óculos'), 'suitcase': ('uma mala', 'mala'),
    'paper': ('um papel', 'papel'), 'document': ('um documento', 'documento'),
}
ENVIRONMENTS = {
    'street': 'em uma rua', 'road': 'em uma estrada',
    'park': 'em um parque', 'forest': 'em uma floresta',
    'beach': 'em uma praia', 'kitchen': 'em uma cozinha',
    'office_room': 'em um escritório', 'office': 'em um escritório',
    'classroom': 'em uma sala de aula', 'bedroom': 'em um quarto',
    'living_room': 'em uma sala de estar', 'restaurant_indoor': 'em um restaurante',
    'restaurant': 'em um restaurante', 'school': 'em uma escola',
    'hospital': 'em um hospital', 'store': 'em uma loja',
    'indoor': 'em ambiente interno', 'outdoor': 'ao ar livre',
}
ENV_PRIORITY = [k for k in ENVIRONMENTS if k not in {'indoor', 'outdoor'}] + ['indoor', 'outdoor']
ATMOSPHERE = {
    'rainy': 'sob chuva', 'snowy': 'sob neve', 'foggy': 'com neblina',
    'cloudy': 'sob céu nublado', 'sunny': 'sob sol forte', 'clear_weather': 'com céu limpo',
    'night': 'à noite', 'day': 'durante o dia', 'dawn_dusk': 'ao amanhecer ou entardecer',
    'low_light': 'com pouca iluminação', 'bright': 'em ambiente iluminado',
}
ATM_GROUPS = (('rainy', 'snowy', 'foggy', 'cloudy', 'sunny', 'clear_weather'),
              ('night', 'dawn_dusk', 'day'), ('low_light', 'bright'))
COLORS = ('black_clothes', 'white_clothes', 'gray_clothes', 'blue_clothes', 'red_clothes')
HAIR = ('long_hair', 'short_hair')


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _scene_id(scene, ordinal):
    context = scene.get('context') or {}
    index = context.get('scene_index', scene.get('scene_index', ordinal))
    return f'scene_{int(index):04d}' if str(index).isdigit() else str(index)


def _confirmed_name(value):
    """Apenas nomes de audio_identity.confirmed_names, nunca candidatos."""
    if not isinstance(value, str):
        return None
    name = value.strip()
    if 1 < len(name) <= 45 and re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ '\-]{0,44}", name):
        return name
    return None


def _name_usable_at_time(name, entity_id, audio_identity, reference_time):
    """Não revelar nomes antes do diálogo que os confirmou.

    Mapas de nomes externos sem segmentos continuam permitidos; o resolver
    interno registra o primeiro segmento verificado quando disponível.
    """
    segments = audio_identity.get('segments') or []
    verified_times = []
    for seg in segments:
        for mention in seg.get('name_mentions') or []:
            if (mention.get('verified') is True and mention.get('entity_id') == entity_id
                    and str(mention.get('name', '')).casefold() == name.casefold()):
                verified_times.append(_num(seg.get('end'), _num(seg.get('start'))))
    if verified_times:
        return min(verified_times) <= reference_time + 1e-3
    # Sem segmentos, o nome confirmado foi explicitamente fornecido fora
    # do fluxo temporal. Não existe tempo de introdução a inferir.
    return not segments


def _pick_context(context_record, priority):
    attrs = context_record.get('confirmed_attributes') or {}
    for label in priority:
        if _num(attrs.get(label)) >= 0.5:
            return label
    return None


def _atmosphere_context(record):
    """No máximo 2 descritores, evitando repetir sol/iluminação/dia."""
    attrs = record.get('confirmed_attributes') or {}
    selected = []
    for group in ATM_GROUPS:
        for label in group:
            if _num(attrs.get(label)) >= .5:
                selected.append(label)
                break
    if not selected:
        return []
    # "sunny" já sugere luz; não repetir "day" e "bright".
    if 'sunny' in selected:
        selected = [x for x in selected if x not in {'day', 'bright'}]
    if 'night' in selected:
        selected = [x for x in selected if x != 'bright']
    return selected[:2]


def _choose_action(entity_id, memory, scene_frames):
    """Só narra ações visualmente observadas NESTA cena."""
    scores = defaultdict(dict)
    for frame in scene_frames:
        fid = frame.get('frame_id')
        for action in frame.get('actions') or []:
            if not isinstance(action, dict) or action.get('entity_id') != entity_id:
                continue
            for pred in action.get('predictions') or []:
                label, value = pred.get('label'), _num(pred.get('score'))
                if label in ACTIONS and value >= 0.65:
                    scores[label][fid] = max(scores[label].get(fid, 0.0), value)
    persisted = {a.get('label') for a in (memory or []) if a.get('persistent')}
    candidates = []
    for label, frame_scores in scores.items():
        best = max(frame_scores.values())
        # Uma evidência isolada precisa ser excepcional; caso contrário, duas.
        strong = len(frame_scores) >= 2 or (label in persisted and best >= 0.82)
        if strong:
            candidates.append((label, sum(frame_scores.values()) / len(frame_scores), len(frame_scores)))
    if not candidates:
        return None
    specific = {'walking': 1, 'running': 2, 'cycling': 2, 'phone_use': 1,
                'reading': 1, 'writing': 1, 'driving': 2}
    candidates.sort(key=lambda x: (x[1], specific.get(x[0], 0), x[2]), reverse=True)
    return candidates[0][0]


def _person_appearance(person_frames):
    by_label = defaultdict(list)
    for frame_person in person_frames:
        for label, value in (frame_person.get('attributes') or {}).items():
            if label in (*SUBJECTS, *PERSON_ATTRS):
                by_label[label].append(_num(value))
    def good(label, threshold):
        vals = by_label.get(label) or []
        return bool(vals) and (sum(vals) / len(vals) >= threshold)
    if good('woman', 0.78):
        subject = 'woman'
    elif good('man', 0.78):
        subject = 'man'
    else:
        subject = 'person'
    attrs = []
    best_hair = max(HAIR, key=lambda x: sum(by_label.get(x, [])) / max(1, len(by_label.get(x, []))))
    if good(best_hair, 0.76):
        attrs.append(best_hair)
    best_color = max(COLORS, key=lambda x: sum(by_label.get(x, [])) / max(1, len(by_label.get(x, []))))
    if good(best_color, 0.75):
        attrs.append(best_color)
    if not attrs and good('dress', 0.82):
        attrs.append('dress')
    return subject, attrs


def _choose_relation(entity_id, memory, scene_frames):
    evidence = defaultdict(dict)
    for f in scene_frames:
        fid = f.get('frame_id')
        for rel in f.get('object_associations') or []:
            if rel.get('entity_id') != entity_id:
                continue
            if not rel.get('accepted', True) or rel.get('relation_kind') == 'spatial_context':
                continue
            label, kind = rel.get('object_label'), rel.get('relation')
            if (label not in OBJECTS or kind not in {'carries', 'holds', 'wears'} or
                    _num(rel.get('object_score')) < .5):
                continue
            score = _num(rel.get('evidence_score', rel.get('score')))
            if score < .7:
                continue
            evidence[(kind, label)][fid] = max(evidence[(kind, label)].get(fid, 0.0), score)
    persistent = {(m.get('relation'), m.get('object_label')) for m in memory if m.get('persistent')}
    candidates = []
    for key, frame_scores in evidence.items():
        high = max(frame_scores.values())
        if len(frame_scores) >= 2 or (key in persistent and high >= .8):
            candidates.append((key, high, len(frame_scores)))
    # 'bag' é ancestral de 'backpack': preferir especificidade quando ambas aceitas.
    labels = {c[0][1] for c in candidates}
    if 'backpack' in labels:
        candidates = [c for c in candidates if c[0][1] not in {'bag', 'handbag'}]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[1], c[2]), reverse=True)
    relation, label = candidates[0][0]
    return {'relation': relation, 'object_label': label, 'score': round(candidates[0][1], 4)}


def build_correlated_scene_facts(spectra_outputs: List[dict], correlation: dict) -> List[dict]:
    """Copia cenas, acrescenta evidências locais; não altera a saída Spectra."""
    frames = correlation.get('frames') or []
    mem_actions = correlation.get('person_actions') or {}
    mem_relations = correlation.get('person_object_relations') or {}
    audio_identity = correlation.get('audio_identity') or {}
    confirmed = audio_identity.get('confirmed_names') or {}
    scene_mem = correlation.get('scenes') or {}
    atmosphere_mem = correlation.get('atmospheres') or {}
    enriched = []
    for ordinal, scene in enumerate(spectra_outputs or []):
        entry = deepcopy(scene)
        sid = _scene_id(entry, ordinal)
        start = _num(entry.get('start_time'), 0)
        end = _num(entry.get('end_time'), start)
        # Pausa sugerida é o instante em que a audiodescrição poderia começar.
        # Na ausência dela, não usar uma informação falada depois do início da cena.
        pause = (entry.get('context') or {}).get('suggested_pause') or {}
        narration_time = _num(pause.get('start'), start) if isinstance(pause, dict) else start
        # Preferir cena real para evitar confundir cortes próximos.
        current_frames = [f for f in frames if str(f.get('scene_id')) == sid
                          and start - 1e-4 <= _num(f.get('timestamp')) <= end + 1e-4]
        by_entity = defaultdict(list)
        for f in current_frames:
            for person in f.get('people') or f.get('persons') or []:
                eid = person.get('entity_id')
                if eid:
                    by_entity[eid].append((f, person))
        people = []
        for eid, pairs in sorted(by_entity.items(), key=lambda x: (-len(x[1]), x[0])):
            subject, appearance = _person_appearance([p for _, p in pairs])
            local_frames = [f for f, _ in pairs]
            action = _choose_action(eid, mem_actions.get(eid), local_frames)
            relation = _choose_relation(eid, mem_relations.get(eid, []), local_frames)
            name = _confirmed_name(confirmed.get(eid))
            if name and not _name_usable_at_time(name, eid, audio_identity, narration_time):
                name = None
            people.append({'entity_id': eid, 'confirmed_name': name,
                           'subject': subject, 'appearance': appearance,
                           'action': action, 'relation': relation,
                           'visible_frame_count': len({f['frame_id'] for f, _ in pairs})})
        scene_name = _pick_context(scene_mem.get(sid) or {}, ENV_PRIORITY)
        atmosphere_name = _atmosphere_context(atmosphere_mem.get(sid) or {})
        facts = {'scene_id': sid, 'people': people, 'environment': scene_name,
                 'atmosphere': atmosphere_name, 'source_frame_ids': [f.get('frame_id') for f in current_frames],
                 'source': 'correlation_v06', 'confirmed_names_only': True}
        entry.setdefault('context', {})['correlation_narrative_v07'] = facts
        enriched.append(entry)
    return enriched


def _article(subject, definite=False):
    if subject == 'woman': return 'A mulher' if definite else 'Uma mulher'
    if subject == 'man': return 'O homem' if definite else 'Um homem'
    return 'A pessoa' if definite else 'Uma pessoa'


def _object_phrase(relation):
    if not relation:
        return None
    label, kind = relation['object_label'], relation['relation']
    prefix = {'carries': 'carrega', 'holds': 'segura', 'wears': 'usa'}.get(kind)
    if not prefix or label not in OBJECTS:
        return None
    return prefix + ' ' + OBJECTS[label][0]


def _build_person_description(person, *, definite=False, include_appearance=True):
    name = person.get('confirmed_name')
    subject = name or _article(person.get('subject'), definite=definite)
    details = []
    if include_appearance:
        details = [PERSON_ATTRS[a] for a in person.get('appearance') or [] if a in PERSON_ATTRS]
    phrase = subject + ((' ' + ' e '.join(details)) if details else '')
    action = person.get('action')
    relation = _object_phrase(person.get('relation'))
    if action in ACTIONS:
        action_text = ACTIONS[action]
        if action == 'sitting' and person.get('subject') == 'man':
            action_text = 'está sentado'
        phrase += ' ' + action_text
        if relation:
            phrase += ' e ' + relation
    elif relation:
        phrase += ' ' + relation
    else:
        phrase += ' aparece'
    return phrase


class CorrelationNarrativeGenerator(LLMNarrativeGenerator):
    """Conserva a lógica de Qwen, filtros de redundância e contrato do TTS."""

    def generate_timeline_from_dicts(self, spectra_outputs, correlation_result=None):
        if not correlation_result:
            return super().generate_timeline_from_dicts(spectra_outputs)
        scenes = build_correlated_scene_facts(spectra_outputs, correlation_result)
        return super().generate_timeline_from_dicts(scenes)

    def _select_facts(self, narrative_input):
        facts = super()._select_facts(narrative_input)
        enriched = (narrative_input.context or {}).get('correlation_narrative_v07')
        # Não sobrepor narrativa antiga se Correlation não ancorou ninguém.
        if not isinstance(enriched, dict):
            return facts
        people = enriched.get('people') or []
        if not people:
            # Usa categorias Scene/Atmosphere temporais quando confirmadas,
            # sem transformar ausência de pessoas em ausência de outros sujeitos.
            if enriched.get('environment'):
                facts['environment'] = enriched['environment']
            for label in enriched.get('atmosphere') or []:
                if label in {'day', 'night', 'dawn_dusk'}: facts['time'] = label
                elif label in {'low_light', 'bright'}: facts['lighting'] = label
                else: facts['atmosphere'] = label
            return facts
        facts['correlated_people'] = people
        facts['correlated_scene_id'] = enriched['scene_id']
        facts['correlated_environment'] = enriched.get('environment') or facts.get('environment')
        facts['correlated_atmosphere'] = enriched.get('atmosphere')
        facts['correlated_source_frames'] = enriched.get('source_frame_ids')
        return facts

    def _build_base_description(self, facts):
        people = facts.get('correlated_people')
        if not people:
            return super()._build_base_description(facts)
        env_label = facts.get('correlated_environment')
        env = ENVIRONMENTS.get(env_label, '')
        atmos = ', '.join(ATMOSPHERE[x] for x in (facts.get('correlated_atmosphere') or []) if x in ATMOSPHERE)
        if len(people) == 1:
            p = people[0]
            phrase = _build_person_description(p)
            if env: phrase += ' ' + env
            if atmos: phrase += ', ' + atmos
            return phrase.rstrip('.') + '.'
        # Não aplicar ações individuais ao conjunto inteiro.
        count = len(people)
        base = ('Duas pessoas' if count == 2 else
                'Três pessoas' if count == 3 else 'Um grupo de pessoas') + ' aparecem'
        if env: base += ' ' + env
        # Menciona 1 fato individual, identificado por nome apenas se confirmado.
        specific = next((p for p in people if p.get('confirmed_name') and (p.get('action') or p.get('relation'))), None)
        if specific is None:
            specific = next((p for p in people if p.get('action') or p.get('relation')), None)
        if specific:
            if specific.get('confirmed_name'):
                base += '; ' + _build_person_description(specific, include_appearance=False)
            else:
                detail = _build_person_description(specific, include_appearance=False)
                # Uma delas, nunca afirmar que as duas executam a ação.
                if detail.startswith('Uma pessoa '):
                    detail = 'Uma delas ' + detail[len('Uma pessoa '):]
                elif detail.startswith('Uma mulher '):
                    detail = 'Uma delas ' + detail[len('Uma mulher '):]
                elif detail.startswith('Um homem '):
                    detail = 'Um deles ' + detail[len('Um homem '):]
                base += '; ' + detail[:1].lower() + detail[1:]
        if atmos: base += ', ' + atmos
        return base.rstrip('.') + '.'

    def _build_delta_description(self, current, previous):
        people = current.get('correlated_people')
        old = (previous or {}).get('correlated_people')
        if not people:
            return super()._build_delta_description(current, previous)
        if not old:
            return self._build_base_description(current)
        def signature(p):
            rel = p.get('relation') or {}
            return (p.get('entity_id'), p.get('confirmed_name'), p.get('action'),
                    rel.get('relation'), rel.get('object_label'))
        before = sorted([signature(p) for p in old])
        after = sorted([signature(p) for p in people])
        if (before == after and current.get('correlated_environment') == previous.get('correlated_environment')
                and current.get('correlated_atmosphere') == previous.get('correlated_atmosphere')):
            return ''
        # Nome/ação atual, sem reapresentar características visuais antigas.
        if len(people) == 1 and len(old) == 1 and people[0]['entity_id'] == old[0]['entity_id']:
            p = people[0]
            phrase = _build_person_description(p, definite=True, include_appearance=False)
            env = ENVIRONMENTS.get(current.get('correlated_environment'), '')
            if env and current.get('correlated_environment') != previous.get('correlated_environment'):
                phrase += ' ' + env
            return phrase.rstrip('.') + '.'
        return self._build_base_description(current)
